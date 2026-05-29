把面试/评测用的测试数据放在这里：

  test/English/test.txt
  test/Chinese/test.txt

格式与 NER/English/validation.txt 相同（每行「词 标签」，句间空行）。
若 test 无标签，推理后加参数: python run_test_eval.py --no-score

本地彩排可暂时用软链接或复制 validation.txt 为 test.txt。

在项目根目录执行:
  python run_test_eval.py

预测会生成在根目录:
  English_pred_task1.txt  English_pred_task2.txt  English_pred_task3.txt
  Chinese_pred_task1.txt  ...
