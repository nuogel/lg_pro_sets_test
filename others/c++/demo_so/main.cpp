//main.cpp
#include "myAPI.h"
#include <iostream>

int main(){
    std::cout << "1 + 1 = " << add(1, 1) << std::endl;
    std::cout << "1 - 1 = " << minus(1, 1) << std::endl;
    return 0;
}


//先编译我们的 myAPI.cpp 文件生成 myAPI.o 目标文件: g++ -c myAPI.cpp
//生成静态库并使用: ar crv libmyAPI.a myAPI.o
