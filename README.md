# spinup_complete
对spinup代码的完善和整理，加入了pytorch版本的trpo以及离散版本的SAC

项目参考Openai Spinup 项目的格式，每个RL算法均是一个单独的文件
alg
-core.py
-alg.py
-alg_train.py
-alg_test.py
便于学习和开发。
另外，本项目使用了spinup的log工具，需要安装mpi插件
pytorch版本 1.11
