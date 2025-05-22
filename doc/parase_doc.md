# 参数解析库用法解析

这里只介绍了部分常用的方法，更多细节请参考`https://github.com/p-ranav/argparse`。

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

```C++
#include <argparse/argparse.hpp>

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("program_name");

  program.add_argument("square")
    .help("display the square of a given integer")
    .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto input = program.get<int>("square");
  std::cout << (input * input) << std::endl;

  return 0;
}
```
运行上述代码，其结果如下所示：
```bash
foo@bar:/home/dev/$ ./main 15
225
```

#### 添加命名参数

这里的命名参数的原文是Optional Arguments，有部分资料将其命名为可选参数，我觉得这是不太严谨的。在本源码中，全部使用命名参数来指代。

命名参数以`-`或者`--`开头，例如`-a`或者`--verbose`。与位置参数不同，命名参数有对应的名称，根据代码规范，一般以`--`作为全称指代，`-`作为简称，用一个英文字母表示。参数用法如下所示

```C++
argparse::ArgumentParser program("test");

program.add_argument("--verbose")
  .help("increase output verbosity")
  .default_value(false)
  .implicit_value(true);

try {
  program.parse_args(argc, argv);
}
catch (const std::exception& err) {
  std::cerr << err.what() << std::endl;
  std::cerr << program;
  std::exit(1);
}

if (program["--verbose"] == true) {
  std::cout << "Verbosity enabled" << std::endl;
}
```

其结果如下所示：
```bash
foo@bar:/home/dev/$ ./main --verbose
Verbosity enabled
```

对命名参数中提到的部分函数的用法进行解析
1. .default_value：默认参数。如果用户未提供参数，那么将使用其作为默认值。
2. .implicit_value：表示该参数是一种标志位，不用提供具体的参数值，只要提供`--verbose`,那么`verbose`就会被设置为true。

这里再提供其他常见的用法
```C++
// 提供简称、全称
program.add_argument("-o", "--output")
   // 设置该参数为必须项，如果不提供，则会抛出error
  .required()
  .help("specify the output file.");

// 为了避免由于缺少必须项，导致函数报错，可以给该参数赋予默认值。
program.add_argument("-o", "--output")
  .default_value(std::string("-"))
  .required()
  .help("specify the output file.");

// 访问没有默认值的参数
// 如果该参数没有默认值，program.present将会返回nullptr,可以基于此进行判断
if (auto fn = program.present("-o")) {
    do_something_with(*fn);
}

```

使用`.nargs`，将多输入作为输入参数，其用法如下所示：
```C++
argparse::ArgumentParser program("main");

program.add_argument("--input_files")
  .help("The list of input files")
  .nargs(2);

try {
  program.parse_args(argc, argv);   // Example: ./main --input_files config.yml System.xml
}
catch (const std::exception& err) {
  std::cerr << err.what() << std::endl;
  std::cerr << program;
  std::exit(1);
}

auto files = program.get<std::vector<std::string>>("--input_files");  // {"config.yml", "System.xml"}
```
这里支持打包为`std::vector`,`std::list`。个人习惯，更喜欢用`std::vector`。如果输入是`int`类型，那就使用`std::vector<int>`。更多的用法示例如下

```C++
argparse::ArgumentParser program("main");

program.add_argument("--query_point")
  .help("3D query point")
  .nargs(3)
  .default_value(std::vector<double>{0.0, 0.0, 0.0})
  .scan<'g', double>();

try {
  program.parse_args(argc, argv); // Example: ./main --query_point 3.5 4.7 9.2
}
catch (const std::exception& err) {
  std::cerr << err.what() << std::endl;
  std::cerr << program;
  std::exit(1);
}

auto query_point = program.get<std::vector<double>>("--query_point");  // {3.5, 4.7, 9.2}
```

此外，`.nargs`还支持范围参数，具体如下：
```C++
// 1~3个参数
program.add_argument("--input_files")
  .nargs(1, 3);  // This accepts 1 to 3 arguments.

// 任意多（甚至0个）的参数 
program.add_argument("--input_files")
  .nargs(argparse::nargs_pattern::any);  // "*" in Python. This accepts any number of arguments including 0.

// 至少一个参数
program.add_argument("--input_files")
  .nargs(argparse::nargs_pattern::at_least_one);  // "+" in Python. This accepts one or more number of arguments.

// 可选参数
program.add_argument("--input_files")
  .nargs(argparse::nargs_pattern::optional);  // "?" in Python. This accepts an argument optionally.

```






#### 添加标志位参数
刚刚在命名参数章节提到的标志位参数有更简单的写发，如下所示
```C++
argparse::ArgumentParser program("test");

program.add_argument("--verbose")
  .help("increase output verbosity")
  .flag();
```
其效果与其相同。

#### 数字类型参数解析规则
这里要提前说明一点，所有的输入参数，默认都是被解析成`std::string`的。如果用户需要输入数字类型的，那可能需要提前设置解析办法。

用户可以通过`.scan<Shape, T>`指定输入数字的类型。如果输入的数据，无法根据其设置的规则，转换成对应的类型，会产生报错。

其大致用法如下：
```C++
program.add_argument("-x")
       .scan<'d', int>();

program.add_argument("scale")
       .scan<'g', double>();
```

`Shape`表示数据的转换方法，`T`表示对应的转换类型。其转换方法与`std::from_chars`大致相同，但不是完全类似。例如十六进制以0x,0X开头，八进制以0开头。
|---------------| -----------------|
| 转换方法 | 解释说明|
|'a'或 'A' |十六进制的浮点数|
|'e'或'E'| 科学计数法的浮点数 如1.78E9|
|'f'或'F'| 固定浮点法的浮点数 如3.1415|
|'g'或'G'| 一般形式（科学计数或固定浮点），会自适应判断|
|'d'| 十进制 |
|'i'| 与`std::from_chars`类似，基数是10 |
|'o'| 无符号的8进制数|
|'u'| 无符号的十进制数|
|'x'或'X'| 无符号的十六进制数|
|-------|---------------------------------|

默认的情况是`i`。整数参数会按 十进制 解析，浮点参数会按 通用格式 解析。


## 高阶用法

### 指定多个输入作为某个参数
```C++
program.add_argument("--color")
  .default_value<std::vector<std::string>>({ "orange" })
  .append()
  .help("specify the cat's fur color");

try {
  program.parse_args(argc, argv);    // Example: ./main --color red --color green --color blue
}
catch (const std::exception& err) {
  std::cerr << err.what() << std::endl;
  std::cerr << program;
  std::exit(1);
}

auto colors = program.get<std::vector<std::string>>("--color");  // {"red", "green", "blue"}

```
这里的colors参数就是`std::vector<std::string>`,包含多个指定的输入。

### 设置互斥组

在互斥组中的参数，最多只能有一个输入，如果同时有多个输入，就会报错。用法如下所示：
```C++
// 这里的true表示必须要有一个参数输入。默认值是false
auto &group = program.add_mutually_exclusive_group(true);
group.add_argument("--first");
group.add_argument("--second");
```

输入如下：
```bash
foo@bar:/home/dev/$ ./main --first 1 --second 2
Argument '--second VAR' not allowed with '--first VAR'

foo@bar:/home/dev/$ ./main
One of the arguments '--first VAR' or '--second VAR' is required
```
如果输入两个参数，那么就会报错！不输入参数也会报错！

### 将输入参数直接绑定到变量

目前支持 `bool`,`int`,`double`,`std::string`,`std::vector<std::string>`,`std::vector<int>`,其用法如下所示：
```C++
bool flagvar = false;
program.add_argument("--flagvar").store_into(flagvar);

int intvar = 0;
program.add_argument("--intvar").store_into(intvar);

double doublevar = 0;
program.add_argument("--doublevar").store_into(doublevar);

std::string strvar;
program.add_argument("--strvar").store_into(strvar);

std::vector<std::string> strvar_repeated;
program.add_argument("--strvar-repeated").append().store_into(strvar_repeated);

std::vector<std::string> strvar_multi_valued;
program.add_argument("--strvar-multi-valued").nargs(2).store_into(strvar_multi_valued);

std::vector<int> intvar_multi_valued;
program.add_argument("--intvar-multi-valued").nargs(2).store_into(intvar_multi_valued);
```