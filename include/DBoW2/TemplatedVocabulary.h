/**
 * File: TemplatedVocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary 
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_TEMPLATED_VOCABULARY__
#define __D_T_TEMPLATED_VOCABULARY__

#include <cassert>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <opencv2/core.hpp>

#include "FeatureVector.h"
#include "BowVector.h"
#include "ScoringObject.h"

namespace DBoW2 {

/// @param TDescriptor class of descriptor
/// @param F class of descriptor functions
template<class TDescriptor, class F>
/// Generic Vocabulary  模板词典类
class TemplatedVocabulary
{		
public:
  
  /**
   * Initiates an empty vocabulary 初始化一个空词典  构造函数  成员变量赋值 创建空词典
   * @param k branching factor
   * @param L depth levels
   * @param weighting weighting type
   * @param scoring scoring type
   */
  TemplatedVocabulary(int k = 10, int L = 5, 
    WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);
  
  /**
   * Creates the vocabulary by loading a file 由文件加载词典
   * @param filename
   */
  TemplatedVocabulary(const std::string &filename);
  
  /**
   * Creates the vocabulary by loading a file由文件加载词典
   * @param filename
   */
  TemplatedVocabulary(const char *filename);
  
  /** 
   * Copy constructor 词典拷贝构造
   * @param voc
   */
  TemplatedVocabulary(const TemplatedVocabulary<TDescriptor, F> &voc);
  
  /**
   * Destructor
   */
  virtual ~TemplatedVocabulary();
  
  /** 
   * Assigns the given vocabulary to this by copying its data and removing
   * all the data contained by this vocabulary before =重载
   * @param voc
   * @return reference to this vocabulary
   */
  TemplatedVocabulary<TDescriptor, F>& operator=(
    const TemplatedVocabulary<TDescriptor, F> &voc);
  
  /** 
   * Creates a vocabulary from the training features with the already
   * defined parameters  由训练特征参数生成词典
   * @param training_features
   */
  virtual void create
    (const std::vector<std::vector<TDescriptor> > &training_features);
  
  /**
   * Creates a vocabulary from the training features, setting the branching
   * factor and the depth levels of the tree 由训练集特征，指定生成树的分支数和深度
   * @param training_features
   * @param k branching factor
   * @param L depth levels
   */
  virtual void create
    (const std::vector<std::vector<TDescriptor> > &training_features, 
      int k, int L);

  /**
   * Creates a vocabulary from the training features, setting the branching
   * factor and the depth levels of the tree, and the weighting and scoring
   * schemes   指定树分支数、深度、权重类型、评分方式
   */
  virtual void create
    (const std::vector<std::vector<TDescriptor> > &training_features,
      int k, int L, WeightingType weighting, ScoringType scoring);

  /**
   * Returns the number of words in the vocabulary
   * @return number of words  返回单词数量的虚函数
   */
  virtual inline unsigned int size() const;
  
  /**
   * Returns whether the vocabulary is empty (i.e. it has not been trained)
   * @return true iff the vocabulary is empty判断词典是否为空
   */
  virtual inline bool empty() const;

  /**
   * Transforms a set of descriptores into a bow vector将一组描述子转化为词袋向量
   * @param features
   * @param v (out) bow vector of weighted words
   */
  virtual void transform(const std::vector<TDescriptor>& features, BowVector &v) 
    const;
  
  /**
   * Transform a set of descriptors into a bow vector and a feature vector 一组描述子转化为词袋向量和特征容器
   * @param features
   * @param v (out) bow vector
   * @param fv (out) feature vector of nodes and feature indexes   节点和节点中特征的索引
   * @param levelsup levels to go up the vocabulary tree to get the node index 词汇树上一层叶子节点索引
   */
  virtual void transform(const std::vector<TDescriptor>& features,
    BowVector &v, FeatureVector &fv, int levelsup) const;

  /**
   * Transforms a single feature into a word (without weight) 特征描述子转化为一个单词
   * @param feature
   * @return word id
   */
  virtual WordId transform(const TDescriptor& feature) const;
  
  /**
   * Returns the score of two vectors两个向量的评分
   * @param a vector
   * @param b vector
   * @return score between vectors
   * @note the vectors must be already sorted and normalized if necessary
   */
  inline double score(const BowVector &a, const BowVector &b) const;
  
  /**
   * Returns the id of the node that is "levelsup" levels from the word given
   * @param wid word id
   * @param levelsup 0..L
   * @return node id. if levelsup is 0, returns the node id associated to the
   *   word id 得到给定单词上一层的节点ID
   */
  virtual NodeId getParentNode(WordId wid, int levelsup) const;
  
  /**
   * Returns the ids of all the words that are under the given node id,
   * by traversing any of the branches that goes down from the node
   * @param nid starting node id
   * @param words ids of words  给定ID的 所有单词
   */
  void getWordsFromNode(NodeId nid, std::vector<WordId> &words) const;
  
  /**
   * Returns the branching factor of the tree (k)得到词典的分支K
   * @return k
   */
  inline int getBranchingFactor() const { return m_k; }
  
  /** 
   * Returns the depth levels of the tree (L)得到词典深度L
   * @return L
   */
  inline int getDepthLevels() const { return m_L; }
  
  /**
   * Returns the real depth levels of the tree on average
   * @return average of depth levels of leaves  树的平均深度
   */
  float getEffectiveLevels() const;
  
  /**
   * Returns the descriptor of a word 返回一个单词对应的一组描述子
   * @param wid word id
   * @return descriptor
   */
  virtual inline TDescriptor getWord(WordId wid) const;
  
  /**
   * Returns the weight of a word单词权重
   * @param wid word id
   * @return weight
   */
  virtual inline WordValue getWordWeight(WordId wid) const;
  
  /** 
   * Returns the weighting method
   * @return weighting method 权重类型
   */
  inline WeightingType getWeightingType() const { return m_weighting; }
  
  /** 
   * Returns the scoring method
   * @return scoring method评分类型
   */
  inline ScoringType getScoringType() const { return m_scoring; }
  
  /**
   * Changes the weighting method
   * @param type new weighting type更改权重类型
   */
  inline void setWeightingType(WeightingType type);
  
  /**
   * Changes the scoring method更改评分类型
   * @param type new scoring type
   */
  void setScoringType(ScoringType type);
  
  /**
   * Saves the vocabulary into a file字典存进文件
   * @param filename
   */
  void save(const std::string &filename) const;
  
  /**
   * Loads the vocabulary from a file 从文件加载字典
   * @param filename
   */
  void load(const std::string &filename);
  
  /** 
   * Saves the vocabulary to a file storage structure字典保存在制定文件结构
   * @param fn node in file storage
   */
  virtual void save(cv::FileStorage &fs, 
    const std::string &name = "vocabulary") const;
  
  /**
   * Loads the vocabulary from a file storage node从制定文件结构u加载字典
   * @param fn first node
   * @param subname name of the child node of fn where the tree is stored.
   *   If not given, the fn node is used instead
   */  
  virtual void load(const cv::FileStorage &fs, 
    const std::string &name = "vocabulary");
  
  /** 
   * Stops those words whose weight is below minWeight.停止这些单词的最小权重
   * Words are stopped by setting their weight to 0. There are not returned
   * later when transforming image features into vectors.单词权重为0时，图像特征转换为向量时不再返回
   * Note that when using IDF or TF_IDF, the weight is the idf part, which
   * is equivalent to -log(f), where f is the frequency of the word  权重是IDF部分 
   * (f = Ni/N, Ni: number of training images where the word is present, 
   * N: number of training images). 训练图像的数量
   * Note that the old weight is forgotten, and subsequent calls to this 
   * function with a lower minWeight have no effect.
   * @return number of words stopped now
   */
  virtual int stopWords(double minWeight);

protected:

  /// Pointer to descriptor  描述子指针
  typedef const TDescriptor *pDescriptor;

  /// Tree node  树节点结构体
  struct Node 
  {
    /// Node id 节点ID
    NodeId id;
    /// Weight if the node is a word 单词权重
    WordValue weight;
    /// Children 子节点ID
    std::vector<NodeId> children;
    /// Parent node (undefined in case of root) 父节点
    NodeId parent;
    /// Node descriptor  节点描述子
    TDescriptor descriptor;

    /// Word id if the node is a word 单词ID
    WordId word_id;

    /**
     * Empty constructor 空构造函数 全部赋值为0
     */
    Node(): id(0), weight(0), parent(0), word_id(0){}
    
    /**
     * Constructor 构造器 添加节点ID
     * @param _id node id
     */
    Node(NodeId _id): id(_id), weight(0), parent(0), word_id(0){}

    /**
     * Returns whether the node is a leaf node
     * @return true iff the node is a leaf如果是叶子节点返回 true
     */
    inline bool isLeaf() const { return children.empty(); }
  };

protected:

  /**
   * Creates an instance of the scoring object accoring to m_scoring  创建评分对象的实例
   */
  void createScoringObject();

  /** 
   * Returns a set of pointers to descriptores
   * @param training_features all the features训练特征描述子
   * @param features (out) pointers to the training features 指针指向每一个训练特征描述子的
   */
  void getFeatures(
    const std::vector<std::vector<TDescriptor> > &training_features,
    std::vector<pDescriptor> &features) const;

  /**
   * Returns the word id associated to a feature 得到一个特征相关的单词ID及权重
   * @param feature
   * @param id (out) word id
   * @param weight (out) word weight
   * @param nid (out) if given, id of the node "levelsup" levels up
   * @param levelsup
   */
  virtual void transform(const TDescriptor &feature, 
    WordId &id, WordValue &weight, NodeId* nid = NULL, int levelsup = 0) const;

  /**
   * Returns the word id associated to a feature得到一个特征相关的单词ID
   * @param feature
   * @param id (out) word id
   */
  virtual void transform(const TDescriptor &feature, WordId &id) const;
      
  /**
   * Creates a level in the tree, under the parent, by running kmeans with
   * a descriptor set, and recursively creates the subsequent levels too 
   * 通过运行带有描述符集的kmeans，在父节点下在树中创建一个级别，然后递归地创建后续级别
   * @param parent_id id of parent node
   * @param descriptors descriptors to run the kmeans on
   * @param current_level current level in the tree  kmeans步骤
   */
  void HKmeansStep(NodeId parent_id, const std::vector<pDescriptor> &descriptors,
    int current_level);

  /**
   * Creates k clusters from the given descriptors with some seeding algorithm.从给定描述子聚成k类
   * @note In this class, kmeans++ is used, but this function should be
   *   overriden by inherited classes.这个函数应该被其他继承类覆盖
   */
  virtual void initiateClusters(const std::vector<pDescriptor> &descriptors,
    std::vector<TDescriptor> &clusters) const;
  
  /**
   * Creates k clusters from the given descriptor sets by running the
   * initial step of kmeans++  开始聚类
   * @param descriptors 
   * @param clusters resulting clusters
   */
  void initiateClustersKMpp(const std::vector<pDescriptor> &descriptors,
    std::vector<TDescriptor> &clusters) const;
  
  /**
   * Create the words of the vocabulary once the tree has been built 树建立后创建单词
   */
  void createWords();
  
  /**
   * Sets the weights of the nodes of tree according to the given features.
   * Before calling this function, the nodes and the words must be already
   * created (by calling HKmeansStep and createWords)根据给定特征设置节点权重
   * @param features
   */
  void setNodeWeights(const std::vector<std::vector<TDescriptor> > &features);
  
  /**
   * Returns a random number in the range [min..max]
   * @param min
   * @param max
   * @return random T number in [min..max]返回随机数
   */
  template <class T>
  static T RandomValue(T min, T max){
      return ((T)rand()/(T)RAND_MAX) * (max - min) + min;
  }

  /**
   * Returns a random int in the range [min..max]
   * @param min
   * @param max
   * @return random int in [min..max]
   */
  static int RandomInt(int min, int max){
      int d = max - min + 1;
      return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
  }

protected:

  /// Branching factor  保护成员变量  分支因子
  int m_k;
  
  /// Depth levels 深度
  int m_L;
  
  /// Weighting method  权重方法
  WeightingType m_weighting;
  
  /// Scoring method 评分方法
  ScoringType m_scoring;
  
  /// Object for computing scores  计算评分对象
  GeneralScoring* m_scoring_object;
  
  /// Tree nodes 树节点容器  存储的是所有节点
  std::vector<Node> m_nodes;
  
  /// Words of the vocabulary (tree leaves)单词  叶子节点
  /// this condition holds: m_words[wid]->word_id == wid
  std::vector<Node*> m_words;
  
};

// ------------------------------------------------------------------类成员函数实现

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary       ///构造函数 成员变量赋值 创建空词典
  (int k, int L, WeightingType weighting, ScoringType scoring)
  : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
  m_scoring_object(NULL)
{
  createScoringObject();
}

// --------------------------------------------------------------------------3个构造函数

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
  (const std::string &filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
  (const char *filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------创建评分对象

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::createScoringObject()
{
  delete m_scoring_object;
  m_scoring_object = NULL;
  
  switch(m_scoring)
  {
    case L1_NORM: 
      m_scoring_object = new L1Scoring;
      break;
      
    case L2_NORM:
      m_scoring_object = new L2Scoring;
      break;
    
    case CHI_SQUARE:
      m_scoring_object = new ChiSquareScoring;
      break;
      
    case KL:
      m_scoring_object = new KLScoring;
      break;
      
    case BHATTACHARYYA:
      m_scoring_object = new BhattacharyyaScoring;
      break;
      
    case DOT_PRODUCT:
      m_scoring_object = new DotProductScoring;
      break;
    
  }
}

// --------------------------------------------------------------------------设置评分类型

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setScoringType(ScoringType type)
{
  m_scoring = type;
  createScoringObject();
}

// --------------------------------------------------------------------------设置权重类型

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setWeightingType(WeightingType type)
{
  this->m_weighting = type;
}

// --------------------------------------------------------------------------拷贝构造函数

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary(
  const TemplatedVocabulary<TDescriptor, F> &voc)
  : m_scoring_object(NULL)
{
  *this = voc;
}

// --------------------------------------------------------------------------析构

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::~TemplatedVocabulary()
{
  delete m_scoring_object;
}

// --------------------------------------------------------------------------赋值运算符重载

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>& 
TemplatedVocabulary<TDescriptor,F>::operator=
  (const TemplatedVocabulary<TDescriptor, F> &voc)
{  
  this->m_k = voc.m_k;
  this->m_L = voc.m_L;
  this->m_scoring = voc.m_scoring;
  this->m_weighting = voc.m_weighting;

  this->createScoringObject();
  
  this->m_nodes.clear();
  this->m_words.clear();
  
  this->m_nodes = voc.m_nodes;
  this->createWords();
  
  return *this;
}

// --------------------------------------------------------------------------由描述子创建词典的步骤

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
  const std::vector<std::vector<TDescriptor> > &training_features)//TDescriptor是一个描述子内部是该一副图片描述子类型的容器 外面是所有图像的描述子
{
  m_nodes.clear();//清空树节点
  m_words.clear();//清空单词容器
  
  // expected_nodes = Sum_{i=0..L} ( k^i )   总节点数计算公式   k=2,l=3    2^4-1/2=7
	int expected_nodes = 
		(int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));

  m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree  提前分配好节点数量，即总节点数量
  
  std::vector<pDescriptor> features;//该函数将输入的所有描述子容器转化为指针容器
  getFeatures(training_features, features);


  // create root  先有根结点
  m_nodes.push_back(Node(0)); // root
  
  // create the tree 递归创建kmeans树
  HKmeansStep(0, features, 1);

  // create the words
  createWords();

  // and set the weight of each node of the tree
  setNodeWeights(training_features);
  
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
  const std::vector<std::vector<TDescriptor> > &training_features,
  int k, int L)
{
  m_k = k;
  m_L = L;
  
  create(training_features);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
  const std::vector<std::vector<TDescriptor> > &training_features,
  int k, int L, WeightingType weighting, ScoringType scoring)
{
  m_k = k;
  m_L = L;
  m_weighting = weighting;
  m_scoring = scoring;
  createScoringObject();
  
  create(training_features);
}

// --------------------------------------------------------------------------将所有描述子容器转化为指针的容器，指针指向每一个描述子

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::getFeatures(
  const std::vector<std::vector<TDescriptor> > &training_features,
  std::vector<pDescriptor> &features) const
{
  features.resize(0);
  
  typename std::vector<std::vector<TDescriptor> >::const_iterator vvit;//外层所有图片的常量迭代器，图片数量
  typename std::vector<TDescriptor>::const_iterator vit;//内层一张图片的描述子迭代器
  for(vvit = training_features.begin(); vvit != training_features.end(); ++vvit)//所有图片迭代
  {
    features.reserve(features.size() + vvit->size());//预留空间 一张图片的描述子个数
    for(vit = vvit->begin(); vit != vvit->end(); ++vit)//内层迭代 一张图片的描述子的迭代 使用箭头获得指针内部
    {
      features.push_back(&(*vit));//容器中存的是指向每个描述子的指针
    }
  }
}

// --------------------------------------------------------------------------kmeans函数 父节点id，描述子指针容器 ，当前层数 -------0 feature 1

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::HKmeansStep(NodeId parent_id, 
  const std::vector<pDescriptor> &descriptors, int current_level)
{
  if(descriptors.empty()) return;//传入描述子为空无法聚类
        
  // features associated to each cluster
  std::vector<TDescriptor> clusters;//聚类容器 一层k个聚类
	std::vector<std::vector<unsigned int> > groups; // groups[i] = [j1, j2, ...]
  // j1, j2, ... indices of descriptors associated to cluster i    与聚类i相关的描述子索引

  clusters.reserve(m_k);//预留空间 一层K个聚类
	groups.reserve(m_k);//k个聚类，存储每个聚类中相关描述子的索引
  
  //const int msizes[] = { m_k, descriptors.size() };
  //cv::SparseMat assoc(2, msizes, CV_8U);
  //cv::SparseMat last_assoc(2, msizes, CV_8U);  
  //// assoc.row(cluster_idx).col(descriptor_idx) = 1 iif associated
  
  if((int)descriptors.size() <= m_k)//传入描述子种类小于K类 则一类一个描述子
  {
    // trivial case: one cluster per feature
    groups.resize(descriptors.size());

    for(unsigned int i = 0; i < descriptors.size(); i++)
    {
      groups[i].push_back(i);
      clusters.push_back(*descriptors[i]);
    }
  }
  else//正常情况 描述子个数>k，聚类
  {
    // select clusters and groups with kmeans
    
    bool first_time = true;
    bool goon = true;
    
    // to check if clusters move after iterations 用来检查聚类中心在迭代后是否移动
    std::vector<int> last_association, current_association;//上一次相关 当前相关

    while(goon)
    {
      // 1. Calculate clusters 计算聚类

			if(first_time)//第一次 随机取k个初始均值 因为要聚成k类  即kmeans++
			{
        // random sample kmeans++方式  执行完后将描述子添加进对应聚类中心
        initiateClusters(descriptors, clusters);
      }
      else//第二次while循环时执行 已经有了k个随机均值 正常kmeans步骤
      {
      // calculate cluster centres 计算聚类中心
      //将描述子添加进对应的类并找到新的均值
        for(unsigned int c = 0; c < clusters.size(); ++c)//size是聚类容器当前的元素数量  此时应为k；c是此层聚类的索引
        {
          std::vector<pDescriptor> cluster_descriptors;//聚类描述子容器，存储属于c聚类的描述子
          cluster_descriptors.reserve(groups[c].size());//第c个聚类相关描述子的个数 
          
          /*
          for(unsigned int d = 0; d < descriptors.size(); ++d)
          {
            if( assoc.find<unsigned char>(c, d) )
            {
              cluster_descriptors.push_back(descriptors[d]);
            }
          }
          */
          
          std::vector<unsigned int>::const_iterator vit;//c聚类中描述子个数常量迭代器
          for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)//c聚类中的描述子个数
          {
            cluster_descriptors.push_back(descriptors[*vit]);//将该描述子指针添加进聚类描述子，表示描述子属于该类
          }
          
          //计算一组描述子的均值  参数为 一组描述子 均值
          F::meanValue(cluster_descriptors, clusters[c]);//把描述子均值存入c中
        }
        
      } // if(!first_time)

      // 2. Associate features with clusters 聚类相关的特征点

      // calculate distances to cluster centers 计算到聚类中心的距离
      groups.clear();
      groups.resize(clusters.size(), std::vector<unsigned int>());//元素数 元素值
      current_association.resize(descriptors.size());

      //assoc.clear();

      typename std::vector<pDescriptor>::const_iterator fit;
      //unsigned int d = 0;
      for(fit = descriptors.begin(); fit != descriptors.end(); ++fit)//, ++d)// 描述子循环 对每一个描述子 寻找到所归属的类
      {
        double best_dist = F::distance(*(*fit), clusters[0]);//F是描述子模板类  调用不同的描述子计算距离的方法  参数是两个描述子 即计算了所有描述子与聚类中心的距离
        unsigned int icluster = 0;
        
        for(unsigned int c = 1; c < clusters.size(); ++c)//聚类的数量 最大为k
        {
          double dist = F::distance(*(*fit), clusters[c]);
          if(dist < best_dist)
          {
            best_dist = dist;//得到距离最近的作为最优
            icluster = c;//该类为此描述子所属类
          }
        }

        //assoc.ref<unsigned char>(icluster, d) = 1;

        groups[icluster].push_back(fit - descriptors.begin());//添加编号 第多少个描述子-起始描述子  迭代器相减是其距离
        current_association[ fit - descriptors.begin() ] = icluster;//此序号的描述子对应的类编号存入容器 
      }
      
      // kmeans++ ensures all the clusters has any feature associated with them

      // 3. check convergence检查聚类 如果聚类中心没有移动，则跳出循环
      if(first_time)
      {
        first_time = false;
      }
      else//不是第一次对该层聚类
      {
        //goon = !eqUChar(last_assoc, assoc);
        
        goon = false;
        for(unsigned int i = 0; i < current_association.size(); i++)
        {
          if([i] != last_association[i]){//聚类中心仍在移动 while循环继续执行
            goon = true;
            break;
          }
        }
      }

			if(goon)
			{
				// copy last feature-cluster association
				last_association = current_association;
				//last_assoc = assoc.clone();
			}
			
		} // while(goon)
    
  } // if must run kmeans
  
  // create nodes 创建该层的k个节点
  for(unsigned int i = 0; i < clusters.size(); ++i)//k个类 k个节点
  {
    NodeId id = m_nodes.size();//节点容器当前的元素个数 即为要输入的id 比如第一次只有根结点 那下一个id就是1
    m_nodes.push_back(Node(id));
    m_nodes.back().descriptor = clusters[i];//.back是最后一个，即当前节点 添加对应描述子 
    m_nodes.back().parent = parent_id;//父节点id，函数输入参数
    m_nodes[parent_id].children.push_back(id);//属于该父节点的子节点id
  }
  
  // go on with the next level 下一层聚类
  if(current_level < m_L)
  {
    // iterate again with the resulting clusters
    const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;//子节点的id
    for(unsigned int i = 0; i < clusters.size(); ++i)//上一层的k个节点 每个节点又要在下一层聚k类 因此每个节点调用一次kmeans
    {
      NodeId id = children_ids[i];

      std::vector<pDescriptor> child_features;
      child_features.reserve(groups[i].size());

      std::vector<unsigned int>::const_iterator vit;
      for(vit = groups[i].begin(); vit != groups[i].end(); ++vit)//上一层第i个聚类中 所包含的描述子 添加后用于下一层的聚类
      {
        child_features.push_back(descriptors[*vit]);
      }

      if(child_features.size() > 1)
      {
        HKmeansStep(id, child_features, current_level + 1);//递归调用
      }
    }
  }
}

// --------------------------------------------------------------------------开始聚类   所有描述子 聚类中心

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::initiateClusters
  (const std::vector<pDescriptor> &descriptors,
   std::vector<TDescriptor> &clusters) const
{
  initiateClustersKMpp(descriptors, clusters);  
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::initiateClustersKMpp(
  const std::vector<pDescriptor> &pfeatures,
    std::vector<TDescriptor> &clusters) const
{
  // Implements kmeans++ seeding algorithm
  // Algorithm:
  // 1. Choose one center uniformly at random from among the data points. 从数据点随机选择一个中心
  // 2. For each data point x, compute D(x), the distance between x and the nearest 
  //    center that has already been chosen.    对于每个数据点x，计算D(x)，即x与最近的（以选的）已选中心之间的距离
  // 3. Add one new data point as a center. Each point x is chosen with probability 
  //    proportional to D(x)^2.                               添加一个新的数据点作为中心。每个点x的选择概率与D(x)^2成比例。
  // 4. Repeat Steps 2 and 3 until k centers have been chosen.  重复步骤2和3，直到k个中心被选择。
  // 5. Now that the initial centers have been chosen, proceed using standard k-means 
  //    clustering.                                                         现在，已经选择了初始中心，继续使用标准k-means聚类。

  clusters.resize(0);
  clusters.reserve(m_k);//重置聚类个数
  std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());//初始化最小距离容器   元素个数 元素初值设为类型的最大值
   
  // 1.随机选了一个描述子的编号
  int ifeature = RandomInt(0, pfeatures.size()-1);
  
  // create first cluster创建第一组聚类
  clusters.push_back(*pfeatures[ifeature]);

  // compute the initial distances
  typename std::vector<pDescriptor>::const_iterator fit;//所有描述子指针迭代器
  std::vector<double>::iterator dit;                                            //迭代器 遍历与聚类中心距离
  dit = min_dists.begin();

  for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)//计算所有描述子到该聚类中心（第一个）的距离
  {
    *dit = F::distance(*(*fit), clusters.back());//fit解一次引用是容器内的描述子指针元素 再解一次则指的是该描述子
  }  

  while((int)clusters.size() < m_k)//未达到聚类个数
  {
    // 2.计算每个数据点与已选中心点（第2，3，，k个）的距离
    dit = min_dists.begin();
    for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)//迭代所有描述子
    {
      if(*dit > 0)//
      {
        double dist = F::distance(*(*fit), clusters.back());
        if(dist < *dit) *dit = dist;//计算出和新聚类中心的距离小于对应之前的聚类中心的距离，则取代
      }
    }
    
    // 3.选择一个新的数据点作为新的聚类中心，选择的原则是：D(x)较大的点，被选取作为聚类中心的概率较大
    double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

    if(dist_sum > 0)
    {
      double cut_d;
      do
      {
        cut_d = RandomValue<double>(0, dist_sum);//返回0-dist_sum之间的非零随机数
      } while(cut_d == 0.0);

      double d_up_now = 0;
      for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
      {
        d_up_now += *dit;//距离求和，求到刚大于cut_d时退出
        if(d_up_now >= cut_d) break;
      }
      
      if(dit == min_dists.end()) //求和求到了最后一个才大于cut_d
        ifeature = pfeatures.size()-1;//取最后一个作为中心
      else//取当前值作为新的聚类中心，他是大距离的概率更大
        ifeature = dit - min_dists.begin();
      
      clusters.push_back(*pfeatures[ifeature]);//把这一个作为聚类中心

    } // if dist_sum > 0
    else
      break;
      
  } // while(used_clusters < m_k)

}

// --------------------------------------------------------------------------创建单词 

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::createWords()
{
  m_words.resize(0);
  
  if(!m_nodes.empty())//节点不为空
  {
    m_words.reserve( (int)pow((double)m_k, (double)m_L) );//预设单词容器大小

    typename std::vector<Node>::iterator nit;//节点迭代器
    
    nit = m_nodes.begin(); // ignore root
    for(++nit; nit != m_nodes.end(); ++nit)//遍历所有节点
    {
      if(nit->isLeaf())//是叶子节点
      {
        nit->word_id = m_words.size();//当前单词数量即为要添加的id 
        m_words.push_back( &(*nit) );//把当前叶子结点的指针添加进单词
      }
    }
  }
}

// --------------------------------------------------------------------------设置单词权重  参数是所有训练描述子

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setNodeWeights
  (const std::vector<std::vector<TDescriptor> > &training_features)
{
  const unsigned int NWords = m_words.size();//单词数量
  const unsigned int NDocs = training_features.size();//训练特征数

  if(m_weighting == TF || m_weighting == BINARY)
  {
    // idf part must be 1 always    IDF部分一直是1
    for(unsigned int i = 0; i < NWords; i++)
      m_words[i]->weight = 1;
  }
  else if(m_weighting == IDF || m_weighting == TF_IDF)
  {
    // IDF and TF-IDF: we calculte the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform  实际上计算的是idf部分，完整tf-idf在transform中计算

    std::vector<unsigned int> Ni(NWords, 0);//容器大小为字典中单词总数 初值为0
    std::vector<bool> counted(NWords, false);
    
    typename std::vector<std::vector<TDescriptor> >::const_iterator mit;
    typename std::vector<TDescriptor>::const_iterator fit;

    for(mit = training_features.begin(); mit != training_features.end(); ++mit)//mit遍历所有训练图片
    {
      fill(counted.begin(), counted.end(), false);//将counted容器中都初始化为false

      for(fit = mit->begin(); fit < mit->end(); ++fit)//遍历一张图的特征 两个循环可以遍历了所有的特征
      {
        WordId word_id;
        transform(*fit, word_id);//传入该特征描述子 转化为对应单词id

        if(!counted[word_id])//该叶子节点在所有训练图像中的出现次数
        {
          Ni[word_id]++;//这个单词的NI++，即特征点数量++
          counted[word_id] = true;
        }
      }
    }

    // set IDF=  ln(N/Ni)  N：所有训练描述子数量   NI:该叶子节点在训练集图像中的出现次数       
    for(unsigned int i = 0; i < NWords; i++)//遍历单词 计算每个单词的权重
    {
      if(Ni[i] > 0)
      {
        m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
      }// else // This cannot occur if using kmeans++
    }
  
  }

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline unsigned int TemplatedVocabulary<TDescriptor,F>::size() const
{
  return m_words.size();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline bool TemplatedVocabulary<TDescriptor,F>::empty() const
{
  return m_words.empty();
}

// --------------------------------------------------------------------------得到树的平均深度

template<class TDescriptor, class F>
float TemplatedVocabulary<TDescriptor,F>::getEffectiveLevels() const
{
  long sum = 0;//所有单词的深度求和
  typename std::vector<Node*>::const_iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); ++wit)//遍历所有单词
  {
    const Node *p = *wit;//指代一个单词
    
    for(; p->id != 0; sum++) //单词id不为0 
      p = &m_nodes[p->parent];  //p指向他的父节点  所以sum为这个单词叶子节点到父节点的深度 
  }
  
  return (float)((double)sum / (double)m_words.size());//sum/单词总数 为平均深度
}

// --------------------------------------------------------------------------得到单词对应的一组描述子

template<class TDescriptor, class F>
TDescriptor TemplatedVocabulary<TDescriptor,F>::getWord(WordId wid) const
{
  return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------得到该单词的权重

template<class TDescriptor, class F>
WordValue TemplatedVocabulary<TDescriptor, F>::getWordWeight(WordId wid) const
{
  return m_words[wid]->weight;
}

// --------------------------------------------------------------------------特征点->单词

template<class TDescriptor, class F>
WordId TemplatedVocabulary<TDescriptor, F>::transform
  (const TDescriptor& feature) const
{
  if(empty())
  {
    return 0;
  }
  
  WordId wid;
  transform(feature, wid);
  return wid;
}

// --------------------------------------------------------------------------一组描述子->词袋向量

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(
  const std::vector<TDescriptor>& features, BowVector &v) const
{
  v.clear();
  
  if(empty())
  {
    return;
  }

  // normalize 
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);

  typename std::vector<TDescriptor>::const_iterator fit;

  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    for(fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w; 
      // w is the idf value if TF_IDF, 1 if TF
      
      transform(*fit, id, w);
      
      // not stopped
      if(w > 0) v.addWeight(id, w);
    }
    
    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) 
        vit->second /= nd;
    }
    
  }
  else // IDF || BINARY
  {
    for(fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY
      
      transform(*fit, id, w);
      
      // not stopped
      if(w > 0) v.addIfNotExist(id, w);
      
    } // if add_features
  } // if m_weighting == ...
  
  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------一组描述子->词袋向量 ，特征容器

template<class TDescriptor, class F> 
void TemplatedVocabulary<TDescriptor,F>::transform(
  const std::vector<TDescriptor>& features,
  BowVector &v, FeatureVector &fv, int levelsup) const
{
  v.clear();
  fv.clear();
  
  if(empty()) // safe for subclasses
  {
    return;
  }
  
  // normalize 
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);
  
  typename std::vector<TDescriptor>::const_iterator fit;
  
  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    unsigned int i_feature = 0;
    for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w; 
      // w is the idf value if TF_IDF, 1 if TF
      
      transform(*fit, id, w, &nid, levelsup);
      
      if(w > 0) // not stopped
      { 
        v.addWeight(id, w);       //词袋：单词id 和权重
        fv.addFeature(nid, i_feature);//节点id和该图像中的特征编号
      }
    }
    
    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) 
        vit->second /= nd;
    }
  
  }
  else // IDF || BINARY
  {
    unsigned int i_feature = 0;
    for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY
      
      transform(*fit, id, w, &nid, levelsup);
      
      if(w > 0) // not stopped
      {
        v.addIfNotExist(id, w);
        fv.addFeature(nid, i_feature);
      }
    }
  } // if m_weighting == ...
  
  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------两个词袋向量的相似性评分

template<class TDescriptor, class F> 
inline double TemplatedVocabulary<TDescriptor,F>::score
  (const BowVector &v1, const BowVector &v2) const
{
  return m_scoring_object->score(v1, v2);
}

// --------------------------------------------------------------------------得到一个特征相关的单词ID

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform
  (const TDescriptor &feature, WordId &id) const  //一个特征  id
{
  WordValue weight;
  transform(feature, id, weight);
}

// --------------------------------------------------------------------------得到一个特征相关的单词ID及权重

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(const TDescriptor &feature, 
  WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
{ 
  // propagate the feature down the tree
  std::vector<NodeId> nodes;
  typename std::vector<NodeId>::const_iterator nit;

  // level at which the node must be stored in nid, if given
  const int nid_level = m_L - levelsup;
  if(nid_level <= 0 && nid != NULL) *nid = 0; // root

  NodeId final_id = 0; // root
  int current_level = 0;

  do
  {
    ++current_level;
    nodes = m_nodes[final_id].children;
    final_id = nodes[0];
 
    double best_d = F::distance(feature, m_nodes[final_id].descriptor);

    for(nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
    {
      NodeId id = *nit;
      double d = F::distance(feature, m_nodes[id].descriptor);
      if(d < best_d)
      {
        best_d = d;
        final_id = id;
      }
    }
    
    if(nid != NULL && current_level == nid_level)
      *nid = final_id;
    
  } while( !m_nodes[final_id].isLeaf() );

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}

// --------------------------------------------------------------------------得到输入单词在某一层的父节点  单词是第0层 

template<class TDescriptor, class F>
NodeId TemplatedVocabulary<TDescriptor,F>::getParentNode
  (WordId wid, int levelsup) const
{
  NodeId ret = m_words[wid]->id; // node id
  while(levelsup > 0 && ret != 0) // ret == 0 --> root
  {
    --levelsup;
    ret = m_nodes[ret].parent;
  }
  return ret;
}

// --------------------------------------------------------------------------得到节点对应的单词   节点id 单词id容器

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::getWordsFromNode
  (NodeId nid, std::vector<WordId> &words) const
{
  words.clear();
  
  if(m_nodes[nid].isLeaf())//该结点是叶子节点
  {
    words.push_back(m_nodes[nid].word_id);//
  }
  else
  {
    words.reserve(m_k); // ^1, ^2, ...
    
    std::vector<NodeId> parents;
    parents.push_back(nid);
    
    while(!parents.empty())
    {
      NodeId parentid = parents.back();
      parents.pop_back();
      
      const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
      std::vector<NodeId>::const_iterator cit;
      
      for(cit = child_ids.begin(); cit != child_ids.end(); ++cit)
      {
        const Node &child_node = m_nodes[*cit];
        
        if(child_node.isLeaf())
          words.push_back(child_node.word_id);
        else
          parents.push_back(*cit);
        
      } // for each child
    } // while !parents.empty
  }
}

// --------------------------------------------------------------------------停止单词 

template<class TDescriptor, class F>
int TemplatedVocabulary<TDescriptor,F>::stopWords(double minWeight)
{
  int c = 0;
  typename std::vector<Node*>::iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); ++wit)//遍历所有单词
  {
    if((*wit)->weight < minWeight)
    {
      ++c;
      (*wit)->weight = 0;
    }
  }
  return c;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::save(const std::string &filename) const
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  save(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::load(const std::string &filename)
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  this->load(fs);
}

// --------------------------------------------------------------------------保存字典

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::save(cv::FileStorage &f,
  const std::string &name) const
{
  // Format YAML:
  // vocabulary 
  // {
  //   k:
  //   L:
  //   scoringType:
  //   weightingType:
  //   nodes 
  //   [
  //     {
  //       nodeId:
  //       parentId:
  //       weight:
  //       descriptor: 
  //     }
  //   ]
  //   words
  //   [
  //     {
  //       wordId:
  //       nodeId:
  //     }
  //   ]
  // }
  //
  // The root node (index 0) is not included in the node vector
  //
  
  f << name << "{";
  
  f << "k" << m_k;
  f << "L" << m_L;
  f << "scoringType" << m_scoring;
  f << "weightingType" << m_weighting;
  
  // tree
  f << "nodes" << "[";
  std::vector<NodeId> parents, children;
  std::vector<NodeId>::const_iterator pit;

  parents.push_back(0); // root

  while(!parents.empty())
  {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node& parent = m_nodes[pid];
    children = parent.children;

    for(pit = children.begin(); pit != children.end(); pit++)
    {
      const Node& child = m_nodes[*pit];

      // save node data
      f << "{:";
      f << "nodeId" << (int)child.id;
      f << "parentId" << (int)pid;
      f << "weight" << (double)child.weight;
      f << "descriptor" << F::toString(child.descriptor);
      f << "}";
      
      // add to parent list
      if(!child.isLeaf())
      {
        parents.push_back(*pit);
      }
    }
  }
  
  f << "]"; // nodes

  // words
  f << "words" << "[";
  
  typename std::vector<Node*>::const_iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); wit++)
  {
    WordId id = wit - m_words.begin();
    f << "{:";
    f << "wordId" << (int)id;
    f << "nodeId" << (int)(*wit)->id;
    f << "}";
  }
  
  f << "]"; // words

  f << "}";

}

// --------------------------------------------------------------------------从文件加载单词

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::load(const cv::FileStorage &fs,
  const std::string &name)
{
  m_words.clear();
  m_nodes.clear();
  
  cv::FileNode fvoc = fs[name];
  
  m_k = (int)fvoc["k"];
  m_L = (int)fvoc["L"];
  m_scoring = (ScoringType)((int)fvoc["scoringType"]);
  m_weighting = (WeightingType)((int)fvoc["weightingType"]);
  
  createScoringObject();

  // nodes
  cv::FileNode fn = fvoc["nodes"];

  m_nodes.resize(fn.size() + 1); // +1 to include root
  m_nodes[0].id = 0;

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId nid = (int)fn[i]["nodeId"];
    NodeId pid = (int)fn[i]["parentId"];
    WordValue weight = (WordValue)fn[i]["weight"];
    std::string d = (std::string)fn[i]["descriptor"];
    
    m_nodes[nid].id = nid;
    m_nodes[nid].parent = pid;
    m_nodes[nid].weight = weight;
    m_nodes[pid].children.push_back(nid);
    
    F::fromString(m_nodes[nid].descriptor, d);
  }
  
  // words
  fn = fvoc["words"];
  
  m_words.resize(fn.size());

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId wid = (int)fn[i]["wordId"];
    NodeId nid = (int)fn[i]["nodeId"];
    
    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];
  }
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary  流操作符重载  写入词典的可打印信息
 * @param os stream to write to 写入流
 * @param voc 词典
 */
template<class TDescriptor, class F>
std::ostream& operator<<(std::ostream &os, 
  const TemplatedVocabulary<TDescriptor,F> &voc)
{
  os << "Vocabulary: k = " << voc.getBranchingFactor() 
    << ", L = " << voc.getDepthLevels()
    << ", Weighting = ";//写入K，L，权重计算方式，评分方式，单词数量

  switch(voc.getWeightingType())
  {
    case TF_IDF: os << "tf-idf"; break;
    case TF: os << "tf"; break;
    case IDF: os << "idf"; break;
    case BINARY: os << "binary"; break;
  }

  os << ", Scoring = ";
  switch(voc.getScoringType())
  {
    case L1_NORM: os << "L1-norm"; break;
    case L2_NORM: os << "L2-norm"; break;
    case CHI_SQUARE: os << "Chi square distance"; break;
    case KL: os << "KL-divergence"; break;
    case BHATTACHARYYA: os << "Bhattacharyya coefficient"; break;
    case DOT_PRODUCT: os << "Dot product"; break;
  }
  
  os << ", Number of words = " << voc.size();

  return os;
}

} // namespace DBoW2

#endif
