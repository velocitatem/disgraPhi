# DisgraPhi Scaling - Quick Reference

## 🚀 Quick Start

### 1. Try the Demo
```bash
# See all scaling features
python examples/scaling_demo.py --demo config

# With your bootstrap model
python examples/scaling_demo.py \
  --bootstrap-adapter ml/models/bootstrap/best_model \
  --demo all
```

### 2. Progressive Training (Easiest to Start)
```bash
# Train user with progressive checkpoints (15, 30, 50 samples)
python ml/models/progressive_trainer.py \
  --bootstrap-adapter ml/models/bootstrap/best_model \
  --user-id alice \
  --action train

# View progress report
python ml/models/progressive_trainer.py \
  --user-id alice \
  --action report
```

### 3. Batch Training (Maximum Throughput)
```bash
# Train all users in parallel
python ml/models/batch_trainer.py \
  --bootstrap-adapter ml/models/bootstrap/best_model \
  --num-parallel 8

# Train specific users
python ml/models/batch_trainer.py \
  --bootstrap-adapter ml/models/bootstrap/best_model \
  --user-ids alice bob charlie \
  --num-parallel 8
```

## 📊 Performance at a Glance

| Approach | Users/Hour | Cost/User | Time to First Model |
|----------|-----------|-----------|---------------------|
| **Sequential** | 10 | $0.50 | 20 min |
| **Batch (8x)** | 70 | $0.06 | N/A |
| **Progressive** | N/A | N/A | **5 min** ⭐ |
| **Both** | 70 | $0.06 | **5 min** 🚀 |

## 🎯 When to Use What

### Use Progressive Training When:
- ✅ You want faster user feedback
- ✅ Users are actively adding samples
- ✅ Engagement is important
- ✅ Starting with small datasets (15-30 samples)

### Use Batch Training When:
- ✅ You have many users to train
- ✅ Cost efficiency is critical
- ✅ You can batch process (e.g., nightly)
- ✅ Maximum GPU utilization matters

### Use Both (Recommended) When:
- ✅ Production deployment
- ✅ Growing user base
- ✅ Want best of both worlds

## ⚙️ Configuration

Edit `ml/config/scaling.yml`:

```yaml
# Adjust parallel jobs based on GPU memory
batch_training:
  num_parallel: 8  # 4 for 8GB GPU, 8 for 16GB, 12 for 24GB

# Customize checkpoints
progressive_training:
  checkpoints:
    - name: "quick_start"
      min_samples: 15  # Adjust thresholds
```

## 🔧 Troubleshooting

### Out of Memory?
```bash
# Reduce parallel jobs
python ml/models/batch_trainer.py --num-parallel 4
```

### Slow Training?
```bash
# Check GPU usage
nvidia-smi

# Increase parallel jobs if GPU <80% utilized
python ml/models/batch_trainer.py --num-parallel 12
```

### Queue Building Up?
```bash
# Add more GPU workers
CUDA_VISIBLE_DEVICES=0 python ml/models/batch_trainer.py --queue-mode &
CUDA_VISIBLE_DEVICES=1 python ml/models/batch_trainer.py --queue-mode &
```

## 📚 Documentation

- **Full Details**: `docs/SCALING_GUIDE.md`
- **Research**: `docs/DATA_COLLECTION_RESEARCH.md`
- **Summary**: `docs/IMPLEMENTATION_SUMMARY.md`

## 💡 Pro Tips

1. **Start with progressive training** - Low risk, immediate user value
2. **Scale gradually** - Test with 3-4 parallel jobs first
3. **Monitor GPU usage** - Optimize `num_parallel` for 80-90% utilization
4. **Combine approaches** - Progressive for active users, batch for bulk processing

## 🎬 Example Workflow

```bash
# 1. User submits 15 samples
# Progressive training kicks in automatically
python ml/models/progressive_trainer.py --user-id alice --action train
# ✓ First model ready in 5 minutes!

# 2. User adds 15 more (30 total)
# Progressive training again
python ml/models/progressive_trainer.py --user-id alice --action train
# ✓ Improved model ready

# 3. Nightly batch processing
# Train all users who reached checkpoints today
python ml/models/batch_trainer.py \
  --bootstrap-adapter ml/models/bootstrap/best_model \
  --num-parallel 8
# ✓ 70 users/hour throughput
```

## 🚦 Production Deployment

```bash
# Option 1: Simple (single GPU)
python ml/models/batch_trainer.py --queue-mode

# Option 2: Docker Compose (multi-GPU)
docker-compose up -d training-worker

# Option 3: Kubernetes (auto-scaling)
kubectl apply -f k8s/training-deployment.yml
```

## ❓ Need Help?

1. Check the troubleshooting section above
2. Review `docs/SCALING_GUIDE.md`
3. Run demo: `python examples/scaling_demo.py --demo all`
4. Open an issue with your config and error logs

---

**Remember**: The best configuration depends on your specific use case. Start conservative, measure, and optimize! 📈
