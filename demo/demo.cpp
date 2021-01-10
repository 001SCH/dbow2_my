/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<vector<float >> > &features);
void changeStructure(const cv::Mat &plain, vector<vector<float>> &out);
void testVocCreation(const vector<vector<vector<float>> > &features);
void testDatabase(const vector<vector<vector<float> > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------
/*
int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);//加载特征描述子

  testVocCreation(features);//生成词典

  wait();

  testDatabase(features);//数据库

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)//注意容器嵌套
{
  features.clear();//stl函数 清除容器对象
  features.reserve(NIMAGES);//预留空间阈值

  cv::Ptr<cv::ORB> orb = cv::ORB::create();//新建ORB对象    cv::Ptr<cv::xfeatures2d::SURF> orb=cv::xfeatures2d::SURF::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);//得到一幅图片的特征点和描述子

    features.push_back(vector<cv::Mat >());//添加一个空的Mat
    changeStructure(descriptors, features.back());//.back返回最后一个元素的引用，用feature存储描述子
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);//容器头指向行头
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L2_NORM;

  OrbVocabulary voc(k, L, weight, scoring);//构造字典

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);//根据描述子聚类生成字典
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);//图像转化为单词并计算评分
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}
*/
// ----------------------------------------------------------------------------

//SURF特征检测的实现
int main()
{
  vector<vector<vector<float>> > features;
  loadFeatures(features);//加载特征描述子

  testVocCreation(features);//生成词典

  wait();

  testDatabase(features);//数据库

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float >> > &features)//注意容器嵌套
{
  features.clear();//stl函数 清除容器对象
  features.reserve(NIMAGES);//预留空间阈值

  cv::Ptr<cv::xfeatures2d::SURF> surf=cv::xfeatures2d::SURF::create();//新建ORB对象    

  cout << "Extracting SURF features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)//依次读入4张图片
  {
    stringstream ss;
    ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    surf->detectAndCompute(image, mask, keypoints, descriptors);//得到一幅图片的特征点和描述子

    features.push_back(vector<vector<float> >());//添加一个空的Mat的描述子
    changeStructure(descriptors, features.back());//.back返回最后一个元素的引用，用feature存储该副图片的描述子
  }
}

// ----------------------------------------------------------------------------将描述子的Mat类型更改为feature存储的vectpor
void changeStructure(const cv::Mat &plain, vector<vector<float>> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);//容器头指向行头
  }
}

// ----------------------------------------------------------------------------创建词典

void testVocCreation(const vector<vector<vector<float>> > &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L2_NORM;

  SurfVocabulary voc(k, L, weight, scoring);//构造空字典

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);//根据描述子聚类生成字典
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary 计算两图相似性评分
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);//图像转化为单词并计算评分
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------数据库

void testDatabase(const vector<vector<vector<float> > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  SurfVocabulary voc("small_voc.yml.gz");
  
  SurfDatabase db(voc, false, 0); // false = do not use direct index由字典生成数据库 不用顺序索引
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database 查询数据库
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added保存数据库
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again   保存之后可以重新加载
  cout << "Retrieving database once again..." << endl;
  SurfDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------
