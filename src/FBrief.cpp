/**
 * File: FBrief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for BRIEF descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FBrief.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------256位的描述子
//Brief均值 返回mean
void FBrief::meanValue(const std::vector<FBrief::pDescriptor> &descriptors, 
  FBrief::TDescriptor &mean)
{
  mean.reset();
  
  if(descriptors.empty()) return;
  
  const int N2 = descriptors.size() / 2;
  
  vector<int> counters(FBrief::L, 0);//构造 元素长度256bit 元素初值0  即256个int在容器内

  vector<FBrief::pDescriptor>::const_iterator it;//描述子迭代器
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FBrief::TDescriptor &desc = **it;
    for(int i = 0; i < FBrief::L; ++i)
    {
      if(desc[i]) counters[i]++;//非零处设为1  双层for循环，将输入的描述子容器的每一位对应求和 001+100=101
    }
  }
  
  for(int i = 0; i < FBrief::L; ++i)
  {
    if(counters[i] > N2) mean.set(i);//该位大于描述子数量的一半，设置均值
  }
  
}

// --------------------------------------------------------------------------
  //两个描述子距离
double FBrief::distance(const FBrief::TDescriptor &a, 
  const FBrief::TDescriptor &b)
{
  return (double)(a^b).count();
}

// --------------------------------------------------------------------------
  
std::string FBrief::toString(const FBrief::TDescriptor &a)
{
  return a.to_string(); // reversed
}

// --------------------------------------------------------------------------
  
void FBrief::fromString(FBrief::TDescriptor &a, const std::string &s)
{
  stringstream ss(s);
  ss >> a;
}

// --------------------------------------------------------------------------

void FBrief::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  
  mat.create(N, FBrief::L, CV_32F);
  
  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < FBrief::L; ++j, ++p)
    {
      *p = (desc[j] ? 1.f : 0.f);
    }
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

