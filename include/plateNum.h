#pragma once
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
using namespace std;
namespace pr{
    class plateNum
    {
    public:
        std::vector<std::string> nameList;
        std::vector<float> confList;
        std::string name ;
        float confidence;
        // std::pair<std::string,float> res;
        int length;
        plateNum() {}
        ~plateNum() {}

    private:

    };
}


