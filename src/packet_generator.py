"""
Personalization packet generator for DisgraPhi.

Generates PDF practice packets with:
- Journal entries or other training content from YAML config
- QR codes at corners for alignment
- Free-format writing areas (no lines)
- Prompt text displayed for user to copy
"""

import io
import json
import hashlib
import yaml
import qrcode
from pathlib import Path
from typing import List, Dict, Optional, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


class PacketGenerator:
    """
    Generates personalized practice packets as PDFs.

    Args:
        user_id: Unique identifier for the user
        config_path: Path to training.yml config file
        page_size: Paper size (default: letter)
    """

    def __init__(
        self,
        user_id: str,
        config_path: str = "ml/data/training.yml",
        page_size=letter
    ):
        self.user_id = user_id
        self.page_size = page_size
        self.config = self._load_config(config_path)
        self.page_manifest: List[Dict[str, Any]] = []

        # Page dimensions
        self.width, self.height = page_size

        # Load margins from config or use defaults
        margins = self.config.get('packet_config', {}).get('page_margins', {})
        self.margin_left = margins.get('left', 54)
        self.margin_right = margins.get('right', 54)
        self.margin_top = margins.get('top', 72)
        self.margin_bottom = margins.get('bottom', 72)

        # Writing area config
        writing_area = self.config.get('packet_config', {}).get('writing_area', {})
        self.min_writing_height = writing_area.get('min_height', 200)
        self.entry_spacing = writing_area.get('spacing_between_entries', 36)

        # QR code size
        self.qr_size = 0.6 * inch

        # Calculate available writing area
        self.writing_width = self.width - self.margin_left - self.margin_right
        self.available_height = (
            self.height - self.margin_top - self.margin_bottom - 2 * self.qr_size
        )

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(
                f"Training config not found: {config_path}\n"
                "Please create ml/data/training.yml"
            )

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def generate(self, output_path: str, task_type: Optional[str] = None) -> Dict[str, str]:
        """
        Generate packet PDF and return ground truth mapping.

        Args:
            output_path: Path to save the generated PDF
            task_type: Type of content to generate (e.g., 'journal_entries', 'story_prompts')
                      If None, uses default from config

        Returns:
            Dictionary mapping entry IDs to expected text
        """
        # Determine content type
        if task_type is None:
            task_type = self.config.get('packet_config', {}).get('default_task', 'journal_entries')

        # Get content entries
        entries = self.config.get(task_type, [])
        if not entries:
            raise ValueError(f"No content found for task type: {task_type}")

        # Limit to configured items per packet
        items_per_packet = self.config.get('packet_config', {}).get('items_per_packet', 5)
        entries = entries[:items_per_packet]

        normalized_entries = [
            {
                'id': entry['id'],
                'prompt': entry['prompt'],
                'text': entry['text'].strip()
            }
            for entry in entries
        ]

        # Create PDF canvas
        c = canvas.Canvas(output_path, pagesize=self.page_size)

        # Pre-compute pagination so we can embed accurate page metadata per QR
        pages = self._paginate_entries(c, normalized_entries)
        if not pages:
            raise ValueError("No entries available to generate packet pages.")

        total_pages = len(pages)
        self.page_manifest = []

        ground_truth = {}
        entry_counter = 1

        for page_index, page in enumerate(pages, start=1):
            if page_index > 1:
                c.showPage()

            if page_index == 1:
                self._add_header(c, task_type)

            page_entry_ids = [item['entry']['id'] for item in page['entries']]
            page_id = self._compute_page_signature(
                entry_ids=page_entry_ids,
                page_number=page_index,
                task_type=task_type
            )

            page_meta = {
                "page_number": page_index,
                "total_pages": total_pages,
                "entry_ids": page_entry_ids,
                "page_id": page_id,
            }
            self.page_manifest.append(page_meta)

            qr_payload = self._build_qr_payload(page_meta)
            self._add_qr_corners(c, qr_payload)

            y_position = self._initial_y_position(page_index)

            for page_entry in page['entries']:
                entry = page_entry['entry']
                y_position = self._add_entry(
                    c,
                    y_position,
                    entry['prompt'],
                    entry['text'],
                    entry_counter
                )
                ground_truth[entry['id']] = entry['text']
                entry_counter += 1

                # Maintain spacing consistency for subsequent entries on the page
                y_position -= self.entry_spacing

        # Add instructions page at the end
        c.showPage()

        # Save PDF
        c.save()

        return ground_truth

    def _add_header(self, c: canvas.Canvas, task_type: str) -> None:
        """Add header to the first page."""
        c.setFont("Helvetica-Bold", 12)
        c.drawString(
            self.width / 2 - 100,
            self.height - 0.5 * inch,
            f"DisgraPhi Practice Packet"
        )

        c.setFont("Helvetica", 9)
        c.drawString(
            self.width / 2 - 80,
            self.height - 0.68 * inch,
            f"User: {self.user_id}"
        )

        # Task description
        task_descriptions = {
            'journal_entries': 'Journal Writing Exercise',
            'story_prompts': 'Creative Writing Exercise',
            'letter_templates': 'Letter Writing Exercise'
        }
        task_desc = task_descriptions.get(task_type, task_type.replace('_', ' ').title())

        c.setFont("Helvetica-Oblique", 8)
        c.drawString(
            self.margin_left,
            self.height - self.margin_top - self.qr_size - 0.3 * inch,
            f"Task: {task_desc}"
        )

    def _add_qr_corners(self, c: canvas.Canvas, payload: Dict[str, Any]) -> None:
        """Add QR codes at the four corners embedding page identification metadata."""
        corner_positions = {
            'TL': (self.margin_left, self.height - self.margin_top - self.qr_size),
            'TR': (self.width - self.margin_right - self.qr_size,
                   self.height - self.margin_top - self.qr_size),
            'BL': (self.margin_left, self.margin_bottom),
            'BR': (self.width - self.margin_right - self.qr_size, self.margin_bottom),
        }

        for corner_id, (x, y) in corner_positions.items():
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=1,
            )
            qr_data = payload.copy()
            qr_data['corner'] = corner_id
            qr.add_data(json.dumps(qr_data, separators=(',', ':'), sort_keys=True))
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Convert PIL image to ReportLab-compatible format
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Draw QR code
            c.drawImage(
                ImageReader(img_buffer),
                x, y,
                width=self.qr_size,
                height=self.qr_size
            )

    def _initial_y_position(self, page_number: int) -> float:
        """Compute the starting Y position for entries on a given page."""
        extra_offset = 0.8 * inch if page_number == 1 else 0.3 * inch
        return self.height - self.margin_top - self.qr_size - extra_offset

    def _paginate_entries(
        self,
        c: canvas.Canvas,
        entries: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Group entries into pages while respecting layout constraints."""
        pages: List[Dict[str, Any]] = []
        current_page_entries: List[Dict[str, Any]] = []
        page_number = 1
        y_position = self._initial_y_position(page_number)

        for idx, entry in enumerate(entries):
            required_height = self._estimate_entry_height(c, entry['prompt'], entry['text'])

            if current_page_entries and y_position - required_height < self.margin_bottom + self.qr_size:
                pages.append(
                    {
                        "page_number": page_number,
                        "entries": current_page_entries,
                    }
                )
                current_page_entries = []
                page_number += 1
                y_position = self._initial_y_position(page_number)

            current_page_entries.append(
                {
                    "entry": entry,
                    "global_index": idx,
                }
            )
            y_position -= required_height + self.entry_spacing

        if current_page_entries:
            pages.append(
                {
                    "page_number": page_number,
                    "entries": current_page_entries,
                }
            )

        return pages

    def _compute_page_signature(
        self,
        entry_ids: List[str],
        page_number: int,
        task_type: str
    ) -> str:
        """Create a compact identifier tying a page to its ground-truth entries."""
        signature_input = f"{self.user_id}|{task_type}|{page_number}|" + '|'.join(entry_ids)
        return hashlib.sha1(signature_input.encode('utf-8')).hexdigest()[:12]

    def _build_qr_payload(
        self,
        page_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble the shared payload embedded in each corner QR."""
        return {
            "page": page_meta["page_id"],
        }

    def _estimate_entry_height(
        self,
        c: canvas.Canvas,
        prompt: str,
        text: str
    ) -> float:
        """Estimate the height needed for an entry."""
        # Prompt header: ~30 points
        # Text display: ~12 points per line, estimate based on text length
        # Writing area: minimum configured height
        # Divider: 10 points

        # Rough estimate: 80 chars per line at font size 9
        text_lines = len(text) // 80 + 1
        text_display_height = text_lines * 12

        total = 30 + text_display_height + 10 + self.min_writing_height + 20
        return total

    def _add_entry(
        self,
        c: canvas.Canvas,
        y_position: float,
        prompt: str,
        text: str,
        entry_num: int
    ) -> float:
        """
        Add a single journal entry with free-format writing area.

        Args:
            c: PDF canvas
            y_position: Current Y coordinate
            prompt: Prompt title (e.g., "Morning Reflection")
            text: Text content to be copied
            entry_num: Entry number for display

        Returns:
            New Y position after adding entry
        """
        start_y = y_position

        # Draw entry number and prompt
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(self.margin_left, y_position, f"{entry_num}. {prompt}")
        y_position -= 20

        # Draw the text to copy (in a box for clarity)
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.3, 0.3, 0.3)

        # Calculate text box dimensions
        text_lines = self._wrap_text(c, text, self.writing_width - 20)
        text_height = len(text_lines) * 12 + 20

        # Draw light background box
        c.setFillColorRGB(0.95, 0.95, 0.95)
        c.rect(
            self.margin_left,
            y_position - text_height,
            self.writing_width,
            text_height,
            fill=1,
            stroke=0
        )

        # Draw text lines
        c.setFillColorRGB(0.3, 0.3, 0.3)
        text_y = y_position - 15
        for line in text_lines:
            c.drawString(self.margin_left + 10, text_y, line)
            text_y -= 12

        y_position -= text_height + 10

        # Add instruction
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(
            self.margin_left,
            y_position - 10,
            "↓ Write this text in your own handwriting below ↓"
        )
        y_position -= 25

        # Draw free-format writing area (just a border, no lines)
        writing_area_height = max(self.min_writing_height, 180)

        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setLineWidth(0.5)
        c.rect(
            self.margin_left,
            y_position - writing_area_height,
            self.writing_width,
            writing_area_height,
            fill=0,
            stroke=1
        )

        y_position -= writing_area_height

        # Draw subtle divider
        y_position -= 10
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.setLineWidth(0.5)
        c.line(
            self.margin_left,
            y_position,
            self.width - self.margin_right,
            y_position
        )

        # Reset colors
        c.setFillColorRGB(0, 0, 0)
        c.setStrokeColorRGB(0, 0, 0)

        return y_position

    def _wrap_text(self, c: canvas.Canvas, text: str, max_width: float) -> List[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            width = c.stringWidth(test_line, "Helvetica", 9)

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _add_instructions_page(
        self,
        c: canvas.Canvas,
        task_type: str,
        num_entries: int
    ) -> None:
        """Add an instructions page at the end."""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(self.margin_left, self.height - 1.5 * inch, "Instructions")

        c.setFont("Helvetica", 10)
        y = self.height - 2 * inch

        instructions = [
            f"1. You have completed a packet with {num_entries} writing exercises.",
            "",
            "2. For each entry:",
            "   • Read the text shown in the gray box",
            "   • Write it in your natural handwriting in the space below",
            "   • Try to write naturally - don't worry about perfection",
            "",
            "3. When you've completed all entries:",
            "   • Photograph each page clearly",
            "   • Make sure all 4 QR codes are visible in each photo",
            "   • Ensure good lighting and the text is readable",
            "",
            "4. Submit your photos using the DisgraPhi processing tool:",
            f"   python ml/data/data.py process-packet --user-id {self.user_id} \\",
            "     --images page1.jpg page2.jpg ... \\",
            "     --ground-truth ground_truth.json",
            "",
            "Tips:",
            "• Write as you normally would - this helps personalize the model",
            "• Take breaks if needed - handwriting quality matters more than speed",
            "• Make sure photos are not blurry or cut off",
        ]

        for line in instructions:
            c.drawString(self.margin_left + 20, y, line)
            y -= 15


def main():
    """CLI interface for packet generation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate DisgraPhi personalization packet"
    )
    parser.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='Unique user identifier'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='packet.pdf',
        help='Output PDF path (default: packet.pdf)'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        default='ground_truth.json',
        help='Output ground truth JSON path'
    )
    parser.add_argument(
        '--task-type',
        type=str,
        choices=['journal_entries', 'story_prompts', 'letter_templates'],
        help='Type of content to generate (default: from config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='ml/data/training.yml',
        help='Path to training config YAML'
    )

    args = parser.parse_args()

    # Generate packet
    generator = PacketGenerator(
        user_id=args.user_id,
        config_path=args.config
    )
    ground_truth = generator.generate(args.output, task_type=args.task_type)

    # Save ground truth
    with open(args.ground_truth, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"✓ Packet generated: {args.output}")
    print(f"✓ Ground truth saved: {args.ground_truth}")
    print(f"  Total entries: {len(ground_truth)}")
    print(f"\nNext steps:")
    print(f"  1. Print {args.output}")
    print(f"  2. Complete all writing exercises")
    print(f"  3. Photograph each page (ensure QR codes are visible)")
    print(f"  4. Run: python ml/data/data.py process-packet --user-id {args.user_id} --images <photos> --ground-truth {args.ground_truth}")


if __name__ == '__main__':
    main()
