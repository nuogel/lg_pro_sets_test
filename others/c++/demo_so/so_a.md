### 生成静态编译库
- 先编译我们的 myAPI.cpp 文件生成 myAPI.o 目标文件:\
  g++ -c myAPI.cpp
- 生成静态库并使用: \
  ar crv libmyAPI.a myAPI.o
- 接下来即可在项目编译过程中利用静态库了，此时 myAPI.cpp 这个库函数的定义文件已经不需要了。 \
  main.cpp 编译命令如下（注意，依赖的静态库文件要放在被依赖项后面）：\
  g++ main.cpp libmyAPI.a -o output 
- 编译通过后即可运行可执行文件 output ， 此时 libmyAPI.a 也已经是不需要的了。执行命令并输出结果如下：\
  ./output

### 生成动态编译库
- linux下编译时通过 -shared 参数可以生成动态库（.so）文件，如下：\
  g++ -shared -fPIC -o libmyAPI.so myAPI.o
- 生成的动态库在编译时需要声明，运行时需要被依赖。声明如下:\
  g++ main.cpp -L. -lmyAPI -o output 
  


