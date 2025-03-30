# 执行基准测试
helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 \
--suite my-suite \
    --max-eval-instances 10

# 汇总和统计
helm-summarize --suite my-suite

# 启动web服务可视化结果，默认端口8000
helm-server --suite my-suite