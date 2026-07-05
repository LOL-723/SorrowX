
#$#
```json
[
  {
    "step_id": "step_1",
    "task": "使用 list_dir 查看 /src/core/tests/tests 目录结构，确认 demo2.py 是否存在",
    "status": "pending",
    "result": null,
    "retry_count": 0
  },
  {
    "step_id": "step_2",
    "task": "如果 demo2.py 存在，使用 read_file 读取其当前内容；如果不存在，则跳过此步",
    "status": "pending",
    "result": null,
    "retry_count": 0
  },
  {
    "step_id": "step_3",
    "task": "使用 write_file 修改或创建 demo2.py，添加代码 print(1+5) 或计算 1+5 并输出的语句",
    "status": "pending",
    "result": null,
    "retry_count": 0
  },
  {
    "step_id": "step_4",
    "task": "使用 run_tests 执行 python demo2.py，验证输出是否为 6",
    "status": "pending",
    "result": null,
    "retry_count": 0
  }
]
```

#$#
```json
[
  {
    "step_id": "step_1",
    "task": "使用 list_dir 列出 /src/core/tests 目录，确认 demo1.py 存在",
    "status": "pending",
    "result": null,
    "retry_count": 0
  },
  {
    "step_id": "step_2",
    "task": "使用 read_file 读取 /src/core/tests/demo1.py 的内容，获取 old_content",
    "status": "pending",
    "result": null,
    "retry_count": 0
  },
  {
    "step_id": "step_3",
    "task": "使用 write_file 将 /src/core/tests/demo1.py 内容修改为 print(1+5)，传入从 step_2 获取的 old_content",
    "status": "pending",
    "result": null,
    "retry_count": 0
  },
  {
    "step_id": "step_4",
    "task": "使用 run_tests 执行 python /src/core/tests/demo1.py，验证输出为 6",
    "status": "pending",
    "result": null,
    "retry_count": 0
  }
]
```
