# 参数解析库用法解析

定义

## 基础用法

### 添加输入参数
首先定义`ArgumentParser`类型的对象，并传入程序名称。当需要添加参数的时候，可以使用`.add_argument`方法。如下所示，这是最基础的用法。

```C++
#include "argparse/argparse.hpp"

argparse::ArgumentParser program("program_name");
program.add_argument("foo");
program.add_argument("-v", "--verbose"); 
```

注意，这里的`foo`是位置参数，`-v`,`--verbose`这种以`-,--`开头的参数是命名参数(Optional Arguments)。

#### 添加位置参数
如下所示，是添加位置参数的一些实例，支持多类型，多数量，下面是一些用法

```bash

```


