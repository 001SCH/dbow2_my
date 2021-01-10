/**
 * File: TemplatedDatabase.h
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated database of images
 * License: see the LICENSE.txt file
 *
 */
 
#ifndef __D_T_TEMPLATED_DATABASE__
#define __D_T_TEMPLATED_DATABASE__

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>

#include "TemplatedVocabulary.h"
#include "QueryResults.h"
#include "ScoringObject.h"
#include "BowVector.h"
#include "FeatureVector.h"

namespace DBoW2 {

// For query functions
static int MIN_COMMON_WORDS = 5;

/// @param TDescriptor class of descriptor
/// @param F class of descriptor functions
template<class TDescriptor, class F>
/// Generic Database  数据库模板类
class TemplatedDatabase
{
public:

  /**
   * Creates an empty database without vocabulary    空字典构造函数
   * @param use_di a direct index is used to store feature indexes 使用顺序索引
   * @param di_levels levels to go up the vocabulary tree to select the  顺序索引中的层数
   *   node id to store in the direct index when adding images
   */
  explicit TemplatedDatabase(bool use_di = true, int di_levels = 0);

  /**
   * Creates a database with the given vocabulary        根据字典构造数据库
   * @param T class inherited from TemplatedVocabulary<TDescriptor, F>
   * @param voc vocabulary
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   */
  template<class T>
  explicit TemplatedDatabase(const T &voc, bool use_di = true, 
    int di_levels = 0);

  /**
   * Copy constructor. Copies the vocabulary too  拷贝构造 
   * @param db object to copy
   */
  TemplatedDatabase(const TemplatedDatabase<TDescriptor, F> &db);

  /** 
   * Creates the database from a file  从文件加载数据库
   * @param filename
   */
  TemplatedDatabase(const std::string &filename);

  /** 
   * Creates the database from a file从文件加载数据库
   * @param filename
   */
  TemplatedDatabase(const char *filename);

  /**
   * Destructor 析构函数
   */
  virtual ~TemplatedDatabase(void);

  /**
   * Copies the given database and its vocabulary 赋值运算符重载拷贝
   * @param db database to copy
   */
  TemplatedDatabase<TDescriptor,F>& operator=(
    const TemplatedDatabase<TDescriptor,F> &db);

  /**
   * Sets the vocabulary to use and clears the content of the database.设置要使用的词汇表并清除数据库的内容。
   * @param T class inherited from TemplatedVocabulary<TDescriptor, F>
   * @param voc vocabulary to copy
   */
  template<class T>
  inline void setVocabulary(const T &voc);
  
  /**
   * Sets the vocabulary to use and the direct index parameters, and clears
   * the content of the database
   * @param T class inherited from TemplatedVocabulary<TDescriptor, F>
   * @param voc vocabulary to copy
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   */
  template<class T>
  void setVocabulary(const T& voc, bool use_di, int di_levels = 0);
  
  /**
   * Returns a pointer to the vocabulary used 返回指向要使用的字典的指针
   * @return vocabulary
   */
  inline const TemplatedVocabulary<TDescriptor,F>* getVocabulary() const;

  /** 
   * Allocates some memory for the direct and inverted indexes  为直接索引和倒排索引分配一些内存
   * @param nd number of expected image entries in the database 预计进入数据库的图像数量
   * @param ni number of expected words per image   每一张图像的预计单词数
   * @note Use 0 to ignore a parameter
   */
  void allocate(int nd = 0, int ni = 0);

  /**
   * Adds an entry to the database and returns its index  添加数据库条目 返回他的索引
   * @param features features of the new entry   新条目的特征
   * @param bowvec if given, the bow vector of these features is returned  如果给了 就返回这些特征对应的词袋向量
   * @param fvec if given, the vector of nodes and feature indexes is returned  如果有，返回节点容器和特征索引
   * @return id of new entry
   */
  EntryId add(const std::vector<TDescriptor> &features,
    BowVector *bowvec = NULL, FeatureVector *fvec = NULL);

  /**
   * Adss an entry to the database and returns its index 
   * @param vec bow vector  词袋向量
   * @param fec feature vector to add the entry. Only necessary if using the
   *   direct index   添加特征容器到新条目 只有顺序索引才需要
   * @return id of new entry
   */
  EntryId add(const BowVector &vec, 
    const FeatureVector &fec = FeatureVector() );

  /**
   * Empties the database 清空数据库
   */
  inline void clear();

  /**
   * Returns the number of entries in the database 返回数据库的条目数  即数据库大小
   * @return number of entries in the database
   */
  inline unsigned int size() const;
  
  /**
   * Checks if the direct index is being used 检查是否使用顺序索引
   * @return true iff using direct index
   */
  inline bool usingDirectIndex() const;
  
  /**
   * Returns the di levels when using direct index 如果使用顺序索引返回其层数
   * @return di levels
   */
  inline int getDirectIndexLevels() const;
  
  /**
   * Queries the database with some features  使用一些特征查询数据库，进行匹配
   * @param features query features  输入查询特征
   * @param ret (out) query results     输出查询结果
   * @param max_results number of results to return. <= 0 means all   要返回结果的最大数量
   * @param max_id only entries with id <= max_id are returned in ret.   要返回结果的最大id
   *   < 0 means all
   */
  void query(const std::vector<TDescriptor> &features, QueryResults &ret,
    int max_results = 1, int max_id = -1) const;
  
  /**
   * Queries the database with a vector  使用归一化的词袋向量查询数据库
   * @param vec bow vector already normalized
   * @param ret results
   * @param max_results number of results to return. <= 0 means all
   * @param max_id only entries with id <= max_id are returned in ret. 
   *   < 0 means all
   */
  void query(const BowVector &vec, QueryResults &ret, 
    int max_results = 1, int max_id = -1) const;

  /**
   * Returns the a feature vector associated with a database entry  返回与数据库条目相关联的 特征容器
   * @param id entry id (must be < size())
   * @return const reference to map of nodes and their associated features in
   *   the given entry
   */
  const FeatureVector& retrieveFeatures(EntryId id) const;

  /**
   * Stores the database in a file 将数据库存到文件
   * @param filename
   */
  void save(const std::string &filename) const;
  
  /**
   * Loads the database from a file 从文件加载数据库
   * @param filename
   */
  void load(const std::string &filename);
  
  /** 
   * Stores the database in the given file storage structure  在给定的文件存储结构存储数据库
   * @param fs
   * @param name node name
   */
  virtual void save(cv::FileStorage &fs, 
    const std::string &name = "database") const;
  
  /** 
   * Loads the database from the given file storage structure  从给定的文件存储结构加载数据库
   * @param fs
   * @param name node name
   */
  virtual void load(const cv::FileStorage &fs, 
    const std::string &name = "database");

protected:
  
  /// Query with L1 scoring  使用L1评分查询数据库
  void queryL1(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with L2 scoring  使用L2评分查询数据库
  void queryL2(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with Chi square scoring
  void queryChiSquare(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with Bhattacharyya scoring
  void queryBhattacharyya(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with KL divergence scoring  
  void queryKL(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with dot product scoring
  void queryDotProduct(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;

protected:

  /* Inverted file declaration 逆序文件声明*/
  
  /// Item of IFRow 逆序文件中存储的内容 条目id---权重
  struct IFPair
  {
    /// Entry id  条目id
    EntryId entry_id;
    
    /// Word weight in this entry  该条目的单词权重
    WordValue word_weight;
    
    /**
     * Creates an empty pair  构造一个空对
     */
    IFPair(){}
    
    /**
     * Creates an inverted file pair  构造一个空的逆序文件对
     * @param eid entry id
     * @param wv word weight
     */
    IFPair(EntryId eid, WordValue wv): entry_id(eid), word_weight(wv) {}
    
    /**
     * Compares the entry ids  输入id与条目id比较 相等返回true
     * @param eid
     * @return true iff this entry id is the same as eid
     */
    inline bool operator==(EntryId eid) const { return entry_id == eid; }
  };
  
  /// Row of InvertedFile 逆序文件的一行 是一个列表 列表中每一个元素是一个IFPair
  typedef std::list<IFPair> IFRow;
  // IFRows are sorted in ascending entry_id order
  
  /// Inverted index  逆序索引文件： 一个单词(叶子节点) 存储有单词对应的图像和权重
  typedef std::vector<IFRow> InvertedFile; 
  // InvertedFile[word_id] --> inverted file of that word
  
  /* Direct file declaration 顺序文件声明*/

  /// Direct index  顺序索引文件：  某一层的一个节点中 存储有相关的特征
  typedef std::vector<FeatureVector> DirectFile;
  // DirectFile[entry_id] --> [ directentry, ... ]

protected:

  /// Associated vocabulary与数据库相关联的词典
  TemplatedVocabulary<TDescriptor, F> *m_voc;
  
  /// Flag to use direct index 使用顺序索引的标志
  bool m_use_di;
  
  /// Levels to go up the vocabulary tree to select nodes to store
  /// in the direct index  顺序索引层数
  int m_dilevels;
  
  /// Inverted file (must have size() == |words|) 逆序文件对象
  InvertedFile m_ifile;
  
  /// Direct file (resized for allocation) 顺序文件对象
  DirectFile m_dfile;
  
  /// Number of valid entries in m_dfile  顺序文件有效条目数量
  int m_nentries;
  
};

// --------------------------------------------------------------------------空数据库构造函数实现 四个成员变量赋值

template<class TDescriptor, class F>
TemplatedDatabase<TDescriptor, F>::TemplatedDatabase
  (bool use_di, int di_levels)
  : m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels), m_nentries(0)
{
}

// --------------------------------------------------------------------------  根据字典构造数据库

template<class TDescriptor, class F>
template<class T>
TemplatedDatabase<TDescriptor, F>::TemplatedDatabase
  (const T &voc, bool use_di, int di_levels)
  : m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels)
{
  setVocabulary(voc);//字典先不存入成员变量 而是给设置函数 最后还是给了 m_voc
  clear();
}

// --------------------------------------------------------------------------拷贝构造数据库

template<class TDescriptor, class F>
TemplatedDatabase<TDescriptor,F>::TemplatedDatabase
  (const TemplatedDatabase<TDescriptor,F> &db)
  : m_voc(NULL)
{
  *this = db;
}

// -------------------------------------------------------------------------- 从文件构造数据库

template<class TDescriptor, class F>
TemplatedDatabase<TDescriptor, F>::TemplatedDatabase
  (const std::string &filename)
  : m_voc(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------从文件构造数据库

template<class TDescriptor, class F>
TemplatedDatabase<TDescriptor, F>::TemplatedDatabase
  (const char *filename)
  : m_voc(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------析构 

template<class TDescriptor, class F>
TemplatedDatabase<TDescriptor, F>::~TemplatedDatabase(void)
{
  delete m_voc;
}

// --------------------------------------------------------------------------赋值运算符重载 复制数据库

template<class TDescriptor, class F>
TemplatedDatabase<TDescriptor,F>& TemplatedDatabase<TDescriptor,F>::operator=
  (const TemplatedDatabase<TDescriptor,F> &db)
{
  if(this != &db)
  {
    m_dfile = db.m_dfile;
    m_dilevels = db.m_dilevels;
    m_ifile = db.m_ifile;
    m_nentries = db.m_nentries;
    m_use_di = db.m_use_di;
    setVocabulary(*db.m_voc);
  }
  return *this;
}

// --------------------------------------------------------------------------添加数据库条目    输入一副图像的描述子  输出词袋向量和特征容器

template<class TDescriptor, class F>
EntryId TemplatedDatabase<TDescriptor, F>::add(
  const std::vector<TDescriptor> &features,
  BowVector *bowvec, FeatureVector *fvec)
{
  BowVector aux;
  BowVector& v = (bowvec ? *bowvec : aux);//如果bowvec不为空则返回给他
  
  if(m_use_di && fvec != NULL)//如果使用顺序索引 且要返回特征容器
  {
    m_voc->transform(features, v, *fvec, m_dilevels); // with features 将该图的描述子转化成 词袋向量，特征容器
    return add(v, *fvec); 
  }
  else if(m_use_di)   //有顺序索引才需要特征容器来存入节点对应的描述子编号
  {
    FeatureVector fv;
    m_voc->transform(features, v, fv, m_dilevels); // with features
    return add(v, fv);
  }
  else if(fvec != NULL)
  {
    m_voc->transform(features, v, *fvec, m_dilevels); // with features  逆序索引需要遍历词袋来添加 单词对应的权重  单词中其实存储的是一系列词袋及该词权重
    return add(v);
  }
  else
  {
    m_voc->transform(features, v); // with features
    return add(v);
  }
}

// ---------------------------------------------------------------------------向数据库添加条目  一幅图像的词袋向量  特征容器
//逆序:单词指向的数据结构是 图像号--权重          顺序：某一层节点指向数据结构是图像号 特征点号
template<class TDescriptor, class F>
EntryId TemplatedDatabase<TDescriptor, F>::add(const BowVector &v,
  const FeatureVector &fv)
{
  EntryId entry_id = m_nentries++;//数据库条目数+1  是要添加的条目id

  BowVector::const_iterator vit;

  if(m_use_di)//使用顺序索引 存的是图像特征
  {
    // update direct file 更新顺序索引存储的特征
    if(entry_id == m_dfile.size())//顺序文件满了  使用pushback存入 自动扩容
    {
      m_dfile.push_back(fv);
    }
    else//序号索引方法存入
    {
      m_dfile[entry_id] = fv;
    }
  }
  
  // update inverted file 更新逆序文件 在对应id的单词中存入该词袋的权重   
  for(vit = v.begin(); vit != v.end(); ++vit)//对词袋向量进行迭代
  {
    const WordId& word_id = vit->first;//单词id
    const WordValue& word_weight = vit->second;//词袋中的单词ID和权重
    
    IFRow& ifrow = m_ifile[word_id];                                  //逆序文件容器的第多少个单词  向该单词的列表中添加该词袋(图像)id和权重
    ifrow.push_back(IFPair(entry_id, word_weight)); //将图像的词袋对应id和权重添加进该单词 即该行    
  }
  
  return entry_id;
}

// --------------------------------------------------------------------------设置词典

template<class TDescriptor, class F>
template<class T>
inline void TemplatedDatabase<TDescriptor, F>::setVocabulary
  (const T& voc)
{
  delete m_voc;
  m_voc = new T(voc);
  clear();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
template<class T>
inline void TemplatedDatabase<TDescriptor, F>::setVocabulary
  (const T& voc, bool use_di, int di_levels)
{
  m_use_di = use_di;
  m_dilevels = di_levels;
  delete m_voc;
  m_voc = new T(voc);
  clear();
}

// --------------------------------------------------------------------------返回辞典

template<class TDescriptor, class F>
inline const TemplatedVocabulary<TDescriptor,F>* 
TemplatedDatabase<TDescriptor, F>::getVocabulary() const
{
  return m_voc;
}

// --------------------------------------------------------------------------清空数据库

template<class TDescriptor, class F>
inline void TemplatedDatabase<TDescriptor, F>::clear()
{
  // resize vectors
  m_ifile.resize(0);
  m_ifile.resize(m_voc->size());
  m_dfile.resize(0);
  m_nentries = 0;
}

// --------------------------------------------------------------------------分配内存

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::allocate(int nd, int ni)
{
  // m_ifile already contains |words| items
  if(ni > 0)
  {
    typename std::vector<IFRow>::iterator rit;
    for(rit = m_ifile.begin(); rit != m_ifile.end(); ++rit)
    {
      int n = (int)rit->size();
      if(ni > n)
      {
        rit->resize(ni);
        rit->resize(n);
      }
    }
  }
  
  if(m_use_di && (int)m_dfile.size() < nd)
  {
    m_dfile.resize(nd);
  }
}

// --------------------------------------------------------------------------返回数据库大小

template<class TDescriptor, class F>
inline unsigned int TemplatedDatabase<TDescriptor, F>::size() const
{
  return m_nentries;
}

// --------------------------------------------------------------------------是否使用顺序索引

template<class TDescriptor, class F>
inline bool TemplatedDatabase<TDescriptor, F>::usingDirectIndex() const
{
  return m_use_di;
}

// --------------------------------------------------------------------------逆序层数

template<class TDescriptor, class F>
inline int TemplatedDatabase<TDescriptor, F>::getDirectIndexLevels() const
{
  return m_dilevels;
}

// --------------------------------------------------------------------------查询 要查询的图像描述子  返回结果 最大值 最大id

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::query(
  const std::vector<TDescriptor> &features,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector vec;
  m_voc->transform(features, vec);//图像描述子转化成词袋向量再查询
  query(vec, ret, max_results, max_id);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::query(
  const BowVector &vec, 
  QueryResults &ret, int max_results, int max_id) const
{
  ret.resize(0);
  
  switch(m_voc->getScoringType())
  {
    case L1_NORM:
      queryL1(vec, ret, max_results, max_id);
      break;
      
    case L2_NORM:
      queryL2(vec, ret, max_results, max_id);
      break;
      
    case CHI_SQUARE:
      queryChiSquare(vec, ret, max_results, max_id);
      break;
      
    case KL:
      queryKL(vec, ret, max_results, max_id);
      break;
      
    case BHATTACHARYYA:
      queryBhattacharyya(vec, ret, max_results, max_id);
      break;
      
    case DOT_PRODUCT:
      queryDotProduct(vec, ret, max_results, max_id);
      break;
  }
}

// --------------------------------------------------------------------------L1查询

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::queryL1(const BowVector &vec, //词袋输入 返回结果  最大结果数量 最大id
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;      //迭代词袋
  typename IFRow::const_iterator rit;//迭代一个单词下的词袋
    
  std::map<EntryId, double> pairs;    //键值对对象  用于存储结果
  std::map<EntryId, double>::iterator pit;//迭代
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)//迭代词袋向量
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;//输入词袋对应的单词权重
        
    const IFRow& row = m_ifile[word_id];//该单词id对应的逆序数据库 即字典中出现过该单词的一系列图像及权重
    
    // IFRows are sorted in ascending entry_id order     IFRow按entry_id升序排序
    
    for(rit = row.begin(); rit != row.end(); ++rit)//遍历该单词下的词袋
    {
      const EntryId entry_id = rit->entry_id;//条目id赋值 即字典中的图像号
      const WordValue& dvalue = rit->word_weight;//字典中的词袋权重
      
      if((int)entry_id < max_id || max_id == -1)//条目id未到设定值
      {
        double value = fabs(qvalue - dvalue) - fabs(qvalue) - fabs(dvalue);//相似性评分公式|v1-v2|-|v1|-|v2|
        
        //pairs是一个map,low_bound返回首个不小于entry_id的迭代器，因此如果map里有这个entry_id, pit指向它，否则指向pairs.end()
        pit = pairs.lower_bound(entry_id);//   因为在db库里entry_id是按顺序存的，所以可以这样查找
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))) //返回一个比较键的函数
        {
          pit->second += value;//评分求和   如果已经有entry_id,累加和
        }
        else      //如果没有，插入此id    将所有的条目都插入 并累计评分和
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));//插入位置  键值对
        }
      }
      
    } // for each inverted row  一个单词中的所有词袋
  } // for each query word 词袋中的每一个单词
	
  // move to vector  移动到查询结果容器中  pairs存储的是查询的结果
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    ret.push_back(Result(pit->first, pit->second));
  }
	
  // resulting "scores" are now in [-2 best .. 0 worst]	
  
  // sort vector in ascending order of score  排序
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector裁减容器
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);
  
  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
  //		for all i | v_i != 0 and w_i != 0 
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  QueryResults::iterator qit;//查询结果迭代器    计算最终的L1相似性评分
  for(qit = ret.begin(); qit != ret.end(); qit++) 
    qit->Score = -qit->Score/2.0;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::queryL2(const BowVector &vec, 
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  //map<EntryId, int> counters;
  //map<EntryId, int>::iterator cit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value = - qvalue * dvalue; // minus sign for sorting trick
        
        pit = pairs.lower_bound(entry_id);
        //cit = counters.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value; 
          //cit->second += 1;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
          
          //counters.insert(cit, 
          //  map<EntryId, int>::value_type(entry_id, 1));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  //cit = counters.begin();
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)//, ++cit)
  {
    ret.push_back(Result(pit->first, pit->second));// / cit->second));
  }
	
  // resulting "scores" are now in [-1 best .. 0 worst]	
  
  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) 
	//		for all i | v_i != 0 and w_i != 0 )
	// (Nister, 2006)
	QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++) 
  {
    if(qit->Score <= -1.0) // rounding error
      qit->Score = 1.0;
    else
      qit->Score = 1.0 - sqrt(1.0 + qit->Score); // [0..1]
      // the + sign is ok, it is due to - sign in 
      // value = - qvalue * dvalue
  }
  
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::queryChiSquare(const BowVector &vec, 
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, std::pair<double, int> > pairs;
  std::map<EntryId, std::pair<double, int> >::iterator pit;
  
  std::map<EntryId, std::pair<double, double> > sums; // < sum vi, sum wi >
  std::map<EntryId, std::pair<double, double> >::iterator sit;
  
  // In the current implementation, we suppose vec is not normalized
  
  //map<EntryId, double> expected;
  //map<EntryId, double>::iterator eit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
        // we move the 4 out
        double value = 0;
        if(qvalue + dvalue != 0.0) // words may have weight zero
          value = - qvalue * dvalue / (qvalue + dvalue);
        
        pit = pairs.lower_bound(entry_id);
        sit = sums.lower_bound(entry_id);
        //eit = expected.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second.first += value;
          pit->second.second += 1;
          //eit->second += dvalue;
          sit->second.first += qvalue;
          sit->second.second += dvalue;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, std::pair<double, int> >::value_type(entry_id,
              std::make_pair(value, 1) ));
          //expected.insert(eit, 
          //  map<EntryId, double>::value_type(entry_id, dvalue));
          
          sums.insert(sit, 
            std::map<EntryId, std::pair<double, double> >::value_type(entry_id,
              std::make_pair(qvalue, dvalue) ));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  sit = sums.begin();
  for(pit = pairs.begin(); pit != pairs.end(); ++pit, ++sit)
  {
    if(pit->second.second >= MIN_COMMON_WORDS)
    {
      ret.push_back(Result(pit->first, pit->second.first));
      ret.back().nWords = pit->second.second;
      ret.back().sumCommonVi = sit->second.first;
      ret.back().sumCommonWi = sit->second.second;
      ret.back().expectedChiScore = 
        2 * sit->second.second / (1 + sit->second.second);
    }
  
    //ret.push_back(Result(pit->first, pit->second));
  }
	
  // resulting "scores" are now in [-2 best .. 0 worst]	
  // we have to add +2 to the scores to obtain the chi square score
  
  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++)
  {
    // this takes the 4 into account
    qit->Score = - 2. * qit->Score; // [0..1]
    
    qit->chiScore = qit->Score;
  }
  
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::queryKL(const BowVector &vec, 
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& vi = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {    
      const EntryId entry_id = rit->entry_id;
      const WordValue& wi = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value = 0;
        if(vi != 0 && wi != 0) value = vi * log(vi/wi);
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // resulting "scores" are now in [-X worst .. 0 best .. X worst]
  // but we cannot make sure which ones are better without calculating
  // the complete score

  // complete scores and move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    EntryId eid = pit->first;
    double value = 0.0;

    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
      const WordValue &vi = vit->second;
      const IFRow& row = m_ifile[vit->first];

      if(vi != 0)
      {
        if(row.end() == find(row.begin(), row.end(), eid ))
        {
          value += vi * (log(vi) - GeneralScoring::LOG_EPS);
        }
      }
    }
    
    pit->second += value;
    
    // to vector
    ret.push_back(Result(pit->first, pit->second));
  }
  
  // real scores are now in [0 best .. X worst]

  // sort vector in ascending order
  // (scores are inverted now --the lower the better--)
  std::sort(ret.begin(), ret.end());

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // cannot scale scores
    
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::queryBhattacharyya(
  const BowVector &vec, QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  //map<EntryId, double> pairs;
  //map<EntryId, double>::iterator pit;
  
  std::map<EntryId, std::pair<double, int> > pairs; // <eid, <score, counter> >
  std::map<EntryId, std::pair<double, int> >::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value = sqrt(qvalue * dvalue);
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second.first += value;
          pit->second.second += 1;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, std::pair<double, int> >::value_type(entry_id,
              std::make_pair(value, 1)));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    if(pit->second.second >= MIN_COMMON_WORDS)
    {
      ret.push_back(Result(pit->first, pit->second.first));
      ret.back().nWords = pit->second.second;
      ret.back().bhatScore = pit->second.first;
    }
  }
	
  // scores are already in [0..1]

  // sort vector in descending order
  std::sort(ret.begin(), ret.end(), Result::gt);

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

}

// ---------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::queryDotProduct(
  const BowVector &vec, QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value; 
        if(this->m_voc->getWeightingType() == BINARY)
          value = 1;
        else
          value = qvalue * dvalue;
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    ret.push_back(Result(pit->first, pit->second));
  }
	
  // scores are the greater the better

  // sort vector in descending order
  std::sort(ret.begin(), ret.end(), Result::gt);

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // these scores cannot be scaled
}

// ---------------------------------------------------------------------------检索特征 返回顺序索引中的id对应的特征容器

template<class TDescriptor, class F>
const FeatureVector& TemplatedDatabase<TDescriptor, F>::retrieveFeatures
  (EntryId id) const
{
  assert(id < size());
  return m_dfile[id];
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::save(const std::string &filename) const
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  save(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::save(cv::FileStorage &fs,
  const std::string &name) const
{
  // Format YAML:
  // vocabulary { ... see TemplatedVocabulary::save }
  // database 
  // {
  //   nEntries: 
  //   usingDI: 
  //   diLevels: 
  //   invertedIndex
  //   [
  //     [
  //        { 
  //          imageId: 
  //          weight: 
  //        }
  //     ]
  //   ]
  //   directIndex
  //   [
  //      [
  //        {
  //          nodeId:
  //          features: [ ]
  //        }
  //      ]
  //   ]

  // invertedIndex[i] is for the i-th word
  // directIndex[i] is for the i-th entry
  // directIndex may be empty if not using direct index
  //
  // imageId's and nodeId's must be stored in ascending order
  // (according to the construction of the indexes)

  m_voc->save(fs);
 
  fs << name << "{";
  
  fs << "nEntries" << m_nentries;
  fs << "usingDI" << (m_use_di ? 1 : 0);
  fs << "diLevels" << m_dilevels;
  
  fs << "invertedIndex" << "[";
  
  typename InvertedFile::const_iterator iit;
  typename IFRow::const_iterator irit;
  for(iit = m_ifile.begin(); iit != m_ifile.end(); ++iit)
  {
    fs << "["; // word of IF
    for(irit = iit->begin(); irit != iit->end(); ++irit)
    {
      fs << "{:" 
        << "imageId" << (int)irit->entry_id
        << "weight" << irit->word_weight
        << "}";
    }
    fs << "]"; // word of IF
  }
  
  fs << "]"; // invertedIndex
  
  fs << "directIndex" << "[";
  
  typename DirectFile::const_iterator dit;
  typename FeatureVector::const_iterator drit;
  for(dit = m_dfile.begin(); dit != m_dfile.end(); ++dit)
  {
    fs << "["; // entry of DF
    
    for(drit = dit->begin(); drit != dit->end(); ++drit)
    {
      NodeId nid = drit->first;
      const std::vector<unsigned int>& features = drit->second;
      
      // save info of last_nid
      fs << "{";
      fs << "nodeId" << (int)nid;
      // msvc++ 2010 with opencv 2.3.1 does not allow FileStorage::operator<<
      // with vectors of unsigned int
      fs << "features" << "[" 
        << *(const std::vector<int>*)(&features) << "]";
      fs << "}";
    }
    
    fs << "]"; // entry of DF
  }
  
  fs << "]"; // directIndex
  
  fs << "}"; // database
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::load(const std::string &filename)
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  load(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedDatabase<TDescriptor, F>::load(const cv::FileStorage &fs,
  const std::string &name)
{ 
  // load voc first
  // subclasses must instantiate m_voc before calling this ::load
  if(!m_voc) m_voc = new TemplatedVocabulary<TDescriptor, F>;
  
  m_voc->load(fs);

  // load database now
  clear(); // resizes inverted file 
    
  cv::FileNode fdb = fs[name];
  
  m_nentries = (int)fdb["nEntries"]; 
  m_use_di = (int)fdb["usingDI"] != 0;
  m_dilevels = (int)fdb["diLevels"];
  
  cv::FileNode fn = fdb["invertedIndex"];
  for(WordId wid = 0; wid < fn.size(); ++wid)
  {
    cv::FileNode fw = fn[wid];
    
    for(unsigned int i = 0; i < fw.size(); ++i)
    {
      EntryId eid = (int)fw[i]["imageId"];
      WordValue v = fw[i]["weight"];
      
      m_ifile[wid].push_back(IFPair(eid, v));
    }
  }
  
  if(m_use_di)
  {
    fn = fdb["directIndex"];
    
    m_dfile.resize(fn.size());
    assert(m_nentries == (int)fn.size());
    
    FeatureVector::iterator dit;
    for(EntryId eid = 0; eid < fn.size(); ++eid)
    {
      cv::FileNode fe = fn[eid];
      
      m_dfile[eid].clear();
      for(unsigned int i = 0; i < fe.size(); ++i)
      {
        NodeId nid = (int)fe[i]["nodeId"];
        
        dit = m_dfile[eid].insert(m_dfile[eid].end(), 
          make_pair(nid, std::vector<unsigned int>() ));
        
        // this failed to compile with some opencv versions (2.3.1)
        //fe[i]["features"] >> dit->second;
        
        // this was ok until OpenCV 2.4.1
        //std::vector<int> aux;
        //fe[i]["features"] >> aux; // OpenCV < 2.4.1
        //dit->second.resize(aux.size());
        //std::copy(aux.begin(), aux.end(), dit->second.begin());
        
        cv::FileNode ff = fe[i]["features"][0];
        dit->second.reserve(ff.size());
                
        cv::FileNodeIterator ffit;
        for(ffit = ff.begin(); ffit != ff.end(); ++ffit)
        {
          dit->second.push_back((int)*ffit); 
        }
      }
    } // for each entry
  } // if use_id
  
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the database
 * @param os stream to write to
 * @param db
 */
template<class TDescriptor, class F>
std::ostream& operator<<(std::ostream &os, 
  const TemplatedDatabase<TDescriptor,F> &db)
{
  os << "Database: Entries = " << db.size() << ", "
    "Using direct index = " << (db.usingDirectIndex() ? "yes" : "no");
  
  if(db.usingDirectIndex())
    os << ", Direct index levels = " << db.getDirectIndexLevels();
  
  os << ". " << *db.getVocabulary();
  return os;
}

// --------------------------------------------------------------------------

} // namespace DBoW2

#endif
