#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include "configparser.h"
#include "_editdistance.h"
//#pragma GCC diagnostic ignored "-Wc++11-extensions"
#define DEBUG 0
//#define endl "\n"
#define all(v) v.begin(),v.end()
using namespace __gnu_pbds;
#define gamma gama
ofstream debug;
int q_gram_count;
int edit_dist_count;
vector<char> iterable;
int strand_length;
int num_underlying_strands;
float remerging_edit_dis_threshold; 
int w;
int l;
int r;
int num_local_iter;
int num_global_iter;
int theta_low;
int theta_high;
float fp_strand_fraction_threshold;
int fp_strand_edit_dis_threshold; 
int cluster_size_threshold;
int remerging_compared_prefix_edit_dis_threshold;
int remerging_compared_prefix_length;
float gamma;

int local_num_anchors;
int local_num_anchor_lists;
int global_num_anchors;
int global_num_anchor_lists;
int freq_representative;

bool sorting_or_pairwise;
bool first_iteration;
bool first_iteration_using_index;
bool print_global_partition;
bool print_global_iterations;
bool print_time_stamps;
bool remove_false_positive_and_small_clusters; 
bool remerge_removed_false_positives;
bool auto_tuning;
bool info_uc;

const int iterable_size = 4;
const int gram_size = 3;
const int block_size = 20;
const int prefix_hashing_length = 11;
const int INF = 1e9;

/*----------------------------
 *This parameter(Q_GRAM_STRAND_SIZE) must be set before the compile time. 
 *All other parameters are global and can be configured from the config file.
 *This value of this parameter for dna_strands length l can be set to -> (l/block_size+2)*(all_q_gram_sequences.size())
 */

#define Q_GRAM_STRAND_SIZE 300

static uint64_t state = 0x4d595df4d0f33173; 
static uint64_t const mltiplr = 6364136223846793005u;
static uint64_t const increment  = 1442695040888963407u; 
uint64_t seed;

int cluster_size;
int max_num_thread;

int total_num_of_strands;
vector<string> pool;
vector<string> all_q_gram_sequences;
vector<vector<int> >q_gram_dict;
vector<string> q_gram_dict_string;
vector<int> temp_gram_string;
vector<vector<string> > underlying_clusters;
vector<vector<int> > curr_clusters;
vector<vector<vector<int> > > temp_clusters; 
vector<vector<int> > clean_temp_clusters;
gp_hash_table<string,vector<int> > global_paritions;
vector<vector<string> > global_anchors;
gp_hash_table<string, vector<int> > fnc;  
vector<int> wrongEle;
pthread_mutex_t mutex1;
//since qram_size is small this will suffice


static uint32_t rotr32(uint32_t x, unsigned r){
    return x >> r | x << (-r & 31);
}
uint32_t pcg32(void){
    uint64_t x = state;
    unsigned count = (unsigned)(x >> 59);
    state = x * mltiplr+ increment;
    x ^= x >> 18;                  
    return rotr32((uint32_t)(x >> 27), count);  // 27 = 32 - 5
}
void pcg32_init(uint64_t seed){
    state = seed + increment;
    (void)pcg32();
}
inline int random_no(){
  return pcg32();
}

void q_gram_seq_generator(int length, string curr){
  if(length == 0){
    all_q_gram_sequences.push_back(curr);
    return;
  }
  else{
    int idx;
    if(curr.size() == 0) idx = 0;
    else idx = find(iterable.begin(),iterable.end(),curr.back())-iterable.begin(); 
    if(DEBUG) assert(idx >= 0 && idx < iterable.size());
    for(int i = idx;i < iterable_size ; ++i){
      curr.push_back(iterable[i]);
      q_gram_seq_generator(length-1 , curr);
      curr.pop_back();
    }
  }
}
inline void q_gram_string(const string & s , int offset , vector<int> &temp_gram_string){
  if(DEBUG) assert(s.size() <= block_size);
  if(s.size() < gram_size) return;
  for(int i = 0, len = all_q_gram_sequences.size() ; i < len ; ++i){
    if(offset*len+i >= Q_GRAM_STRAND_SIZE){
      cout << "Change the variable Q_GRAM_STRAND_SIZE to: " << 2*strand_length*all_q_gram_sequences.size()/block_size << endl;
      exit(EXIT_FAILURE);
    }
    
    temp_gram_string[offset*len+i] = s.find(all_q_gram_sequences[i]) + 1;
    
    if(temp_gram_string[offset*len+i] == string::npos)
      temp_gram_string[offset*len+i] = 0;
  }
}
inline void q_gram_sequence(int idx){
  string s = pool[idx];
  vector<int> temp_gram_string(Q_GRAM_STRAND_SIZE);
  string substr_block(block_size , ' ');
  int i, j = 0, cnt = 0, len;
  for(i = 0, len= s.length();  i < len ; ++i ){
    substr_block[j++] = s[i];
    if(j == block_size){
      q_gram_string(substr_block , cnt , temp_gram_string); 
      j = 0;
      ++cnt;
    }
  }
  if(j != 0){
    substr_block.resize(j);
    q_gram_string(substr_block, cnt , temp_gram_string);
  }
  assert(idx < pool.size());
  q_gram_dict[idx] = temp_gram_string;
}

void *q_gram_thread(void *args){
  pair<int, int> *info = static_cast<pair<int,int> *>(args);
  int start_index = (*info).first , length = (*info).second;
  for(int i = start_index , no = 0; no < length ; ++no , ++i){
    q_gram_sequence(i);
  }
  pthread_exit(NULL);
}
void create_q_gram_dict(){
  if(pool.size() == 0) return;
  int num_thread = min(max_num_thread , (int)pool.size());
  if(num_thread == 0){
    cout << "max_num_thread is set to zero" << endl;
    exit(EXIT_FAILURE);
  }
  int each = pool.size()/num_thread;
  int extra = pool.size()%num_thread;
  pair<int,int> info_thread[num_thread];
  pthread_t my_thread[num_thread];
  int curr = 0;
  for(int i = 0;i < num_thread ; ++i){
    info_thread[i] = make_pair(curr , each+(i<extra)); 
    curr += each+(i<extra);
    int res=pthread_create(&my_thread[i],NULL, &q_gram_thread,&info_thread[i]);
    if(res != 0){
      cout << "Error: p_thread create failed" << endl;
      exit(EXIT_FAILURE);
    }
  }
  for(int i = 0;i < num_thread ; ++i)
    pthread_join(my_thread[i] , NULL);
}

int edit_distance(const string &s1, const string &s2){
  int len1 = s1.length() , len2 = s2.length();
  int dp[2][len1+1]; 
  for(int i = 0;i <= len1 ; ++i){
    dp[0][i] = i;
    dp[1][i] = INF;
  }
  for(int i = 1;i <= len2 ; ++i){
    int curr = (i&1), prev = (curr^1);
    dp[curr][0] = i;
    for(int j = 1; j <= len1 ; ++j){
      if(s2[i-1] == s1[j-1]) dp[curr][j] = dp[prev][j-1]; 
      else{
        dp[curr][j] = min(1+dp[prev][j] , 1+dp[prev][j-1]);
        dp[curr][j] = min(dp[curr][j] , 1+dp[curr][j-1]);
      }
    }
  }
  return dp[len2&1][len1];
}
string random_string(int length){
  string ans(length , ' ');
  for(int i = 0;i < length ; ++i)
    ans[i] = iterable[random_no()%iterable.size()];
  return ans;
}
int hamming_distance(int i , int j){
  int dist = 0;
  for (int k=0; k<q_gram_dict[i].size(); k++)
    dist += abs(q_gram_dict[i][k] - q_gram_dict[j][k]);
  return dist;
}
int compute_distance(int i , int j , string type){
  if(DEBUG) assert(i< total_num_of_strands && i >= 0);
  if(DEBUG) assert(j< total_num_of_strands && j >= 0);
  if(type == "edit"){
    //if(DEBUG) assert(dis <= strand_length);
    return eval_edit_distance(pool[i] , pool[j]);
  }
  else if(type == "hamming"){
    //if(DEBUG) assert(dis <= Q_GRAM_STRAND_SIZE);
    return hamming_distance(i,j);
  }
  else{
    assert(0);
  }
}
void create_anchors(string type , vector<vector<string> > &anchors){
  if(DEBUG) assert(type == "global" || type == "local");
  int rows = ((type == "global")? global_num_anchor_lists : local_num_anchor_lists);  
  int cols = ((type == "global")? global_num_anchors  : local_num_anchors);
  anchors.resize(rows , vector<string>(cols));
  for(int i = 0;i < rows ; ++i){
    for(int j = 0;j < cols ; ++j){
      anchors[i][j] = random_string(w);
      assert(anchors[i][j].size() == w);
    }
  }
}
inline string getHashValue(const string &s, const vector<vector<string> > &anchors){
  string hash_value;
  int len = s.size();
  if(DEBUG) assert(anchors.size() == global_num_anchor_lists || anchors.size() == local_num_anchor_lists);
  for(int i = 0 , rows = anchors.size() ; i < rows ; ++i){
    int pos = -1;
    for(int j = 0, cols = anchors[i].size() ; j < cols ; ++j){
      if(pos == -1)
        pos = s.substr(0 , len-w-l).find(anchors[i][j]); 
      else
        break;
    }
    if(pos == -1) continue;
    hash_value += s.substr(pos , w+l);
  }
  return hash_value;
}

void *global_hashing_thread(void *args){
  pair<int,int> *info = static_cast<pair<int,int> *>(args);
  int start_index = (*info).first , length = (*info).second;
  gp_hash_table<string , vector<int> > global_thread_paritions;
  for(int i = start_index , no = 0; no < length ; ++no , ++i){
    int representative=curr_clusters[i][random_no()%curr_clusters[i].size()]; 
    if(first_iteration && first_iteration_using_index){
      global_thread_paritions[pool[representative].substr(0,prefix_hashing_length)].push_back(i);  
    }
    else{
      global_thread_paritions[getHashValue(pool[representative],global_anchors)].push_back(i);
    }
  }
  pthread_mutex_lock(&mutex1);
  for(auto ele : global_thread_paritions)
    global_paritions[ele.first].insert(global_paritions[ele.first].end() , all(ele.second));
  pthread_mutex_unlock(&mutex1);
  pthread_exit(NULL);
}
void global_hashing(){
  if(curr_clusters.size() == 0) return;
  global_paritions.clear();
  if(!first_iteration_using_index || !first_iteration)
    create_anchors("global", global_anchors);
  int num_thread = min(max_num_thread , (int)curr_clusters.size());
  if(num_thread == 0){
    cout << "max_num_thread set to zero" << endl;
    exit(EXIT_FAILURE);
  }
  int each = curr_clusters.size()/num_thread;
  int extra = curr_clusters.size()%num_thread;
  pair<int,int> info_thread[num_thread];
  pthread_t my_thread[num_thread];
  pthread_mutex_init(&mutex1,NULL);
  int curr = 0;
  for(int i = 0;i < num_thread ; ++i){
    info_thread[i] = make_pair(curr , each+(i<extra));
    curr += each + (i < extra);
    int res = pthread_create(&my_thread[i] , NULL, global_hashing_thread,&info_thread[i]);
    if(res != 0){
      cout << "Error: p_thread create failed" << endl;
      exit(EXIT_FAILURE);
    }
  }
  for(int i = 0 ;i < num_thread ; ++i)
    pthread_join(my_thread[i] , NULL);
}

struct hash_struct{
  string hash_value;
  int rep;
  vector<int> cluster;
};
int randFnc(int j){
  return random_no()%j;
}
bool hash_sort(const hash_struct &x , const hash_struct &y){
  return x.hash_value < y.hash_value;
}

void local_clustering(vector<vector<int> > &clusters){
  if(DEBUG) assert(clusters.size() != 0);
  if(clusters.size() == 1) return;
  vector<hash_struct> hash_values(clusters.size());
  vector<vector<string> > anchors;
  for(int k=0;k<num_local_iter;k++){
    create_anchors("local" , anchors);
    for(int i = 0,len = hash_values.size();i < len ; ++i){
      if(k%freq_representative == 0){
        if(k==0 && sorting_or_pairwise) hash_values[i].cluster = clusters[i];
        else if(k == 0) hash_values[i].cluster.clear();
        if(sorting_or_pairwise) hash_values[i].rep = hash_values[i].cluster[random_no()%hash_values[i].cluster.size()]; 
        else hash_values[i].rep = clusters[i][random_no()%clusters[i].size()];
      }
      hash_values[i].hash_value = getHashValue(pool[hash_values[i].rep] , anchors);
    }
    if(sorting_or_pairwise)
      sort(hash_values.begin(),hash_values.end(),hash_sort);
    int i=0 ,j ,n = hash_values.size();
    while(i<n-1){
      j = i+1; 
      while(j < n){
        //if(sorting_or_pairwise) assert(j-i == 1);
        if(!sorting_or_pairwise && hash_values[i].hash_value != hash_values[j].hash_value){
          ++j;
          continue;
        }
        int rep1=hash_values[i].rep;
        int rep2=hash_values[j].rep;
        int hamming_dist=compute_distance(rep1,rep2,"hamming");
        if(hamming_dist<=theta_low){
          if(sorting_or_pairwise){
            hash_values[i].cluster.reserve(hash_values[i].cluster.size() + hash_values[j].cluster.size());
            hash_values[i].cluster.insert(hash_values[i].cluster.end() , hash_values[j].cluster.begin() , hash_values[j].cluster.end());
            hash_values[i].rep = hash_values[i].cluster[random_no()%hash_values[i].cluster.size()];
          }
          else{
            clusters[i].reserve(clusters[i].size() + clusters[j].size());
            clusters[i].insert(clusters[i].end(),clusters[j].begin(),clusters[j].end());
            hash_values[i].rep = clusters[i][random_no()%clusters[i].size()];
            clusters.erase(clusters.begin()+j);
          }
          hash_values.erase(hash_values.begin()+j);
          n--;
        }
        else if(hamming_dist<=theta_high)
        {
          int edit_dist=compute_distance(rep1,rep2,"edit");
          if(edit_dist<=r){
            if(sorting_or_pairwise){
              hash_values[i].cluster.reserve(hash_values[i].cluster.size() + hash_values[j].cluster.size());
              hash_values[i].cluster.insert(hash_values[i].cluster.end() , hash_values[j].cluster.begin() , hash_values[j].cluster.end());
              hash_values[i].rep = hash_values[i].cluster[random_no()%hash_values[i].cluster.size()];
            }
            else{
              clusters[i].reserve(clusters[i].size() + clusters[j].size());
              clusters[i].insert(clusters[i].end(),clusters[j].begin(),clusters[j].end());
              hash_values[i].rep = clusters[i][random_no()%clusters[i].size()];
              clusters.erase(clusters.begin()+j);
            }
            hash_values.erase(hash_values.begin()+j);
            n--;
          }
          else{
            if(sorting_or_pairwise) break;
            else ++j;
          }
        }
        else{
          if(sorting_or_pairwise) break;
          else ++j;
        }
      }
      ++i;
    }
  }
  if(sorting_or_pairwise){
    clusters.resize(hash_values.size());
    for(int i = 0, len = hash_values.size(); i < len ; ++i)
      clusters[i] = hash_values[i].cluster;
  }
  if(DEBUG) assert(clusters.size() >= 1);
  //cout << clusters.size() << " " <<  "done" << endl;
  return;
}

vector<int> perm;
void *thread_task_assigner(void *args){
  pair<int,int> *info = static_cast<pair<int,int> *>(args);
  int start_index = (*info).first , length = (*info).second;
  for(int i = start_index , no = 0; no < length ; ++no , ++i){
    assert(i < temp_clusters.size());
    local_clustering(temp_clusters[perm[i]]);
  }
  pthread_mutex_lock(&mutex1);
  for(int i = start_index , no =0; no < length ; ++no,++i){
    curr_clusters.reserve(curr_clusters.size() + temp_clusters[perm[i]].size());
    curr_clusters.insert(curr_clusters.end(),all(temp_clusters[perm[i]]));
  }
  pthread_mutex_unlock(&mutex1);
  pthread_exit(NULL);
}


void cluster_paritions(){
  if(global_paritions.size() == 0) return;
  //cout << global_paritions.size() << endl;
  perm.resize(global_paritions.size());
  iota(perm.begin() , perm.end() , 0);
  shuffle(perm.begin(), perm.end(), std::mt19937{std::random_device{}()}); 
  temp_clusters.resize(global_paritions.size()); 
  int id = 0;
  for(auto ele : global_paritions){
    if(DEBUG) assert(ele.second.size() != 0);
    temp_clusters[id].resize(ele.second.size());
    int cnt = 0;
    for(auto indexes: ele.second){
      temp_clusters[id][cnt++] = curr_clusters[indexes];
    }
    ++id;
  }
  int num_thread = min(max_num_thread , (int)global_paritions.size());
  if(num_thread == 0){
    cout << "ERROR: max_num_thread set to zero" << endl;
    exit(EXIT_FAILURE);
  }
  pthread_t my_thread[num_thread];
  pthread_mutex_init(&mutex1,NULL);
  int each = global_paritions.size()/num_thread; 
  int extra = global_paritions.size()%num_thread;
  pair<int,int> info_thread[num_thread];
  int curr = 0;
  curr_clusters.clear();
  for(int i = 0 ;i < num_thread ; ++i){
    info_thread[i] = make_pair(curr , each+(i < extra));
    curr += each+(i <extra);
    int res = pthread_create(&my_thread[i] , NULL , &thread_task_assigner , &info_thread[i]); 
    if(res != 0){
      cout << "Error: p_thread create failed" << endl;
      exit(EXIT_FAILURE);
    }
  }
  for(int i = 0;i < num_thread ; ++i)
    pthread_join(my_thread[i] , NULL);
}
 
vector<vector<int> > clustering(const vector<vector<int> > &clusters){
  if(clusters.size() == 0){
    return clusters;
  }
  curr_clusters = clusters;
  float epsilon=4.0/65000;
  int my_count=0;
  int prev=curr_clusters.size(),curr;
  int i;
  for(i = 0;i < num_global_iter; ++i){
    if(print_global_iterations){
      cout << "Global iter :" << i+1<<" #clusters :"<<curr_clusters.size() << endl;
    }
    first_iteration = (i==0);
    global_hashing();
    cluster_paritions();
    curr=curr_clusters.size();
    float curr_val=prev-curr;
    curr_val/=prev;
    if(DEBUG && ((i%10)==0))
      cout<<i+1<<" "<<curr<<" "<<prev<<" "<<curr_val<<" "<<epsilon<<endl;
    if(curr_val<epsilon){
      my_count++;
      if(my_count==3){
        break;
      }
    }
    prev=curr;
  }
  cout << "Num of global_iterations done: " << i << endl;
  return curr_clusters;
}
void *cluster_cleaner(void *args){
  pair<int,int> *info = static_cast<pair<int,int> *>(args);
  int start_index = (*info).first , length = (*info).second;
   vector<vector<int> > personal_clusters;
   vector<int> pers_wrongEle;
   personal_clusters.reserve(length);
   for(int i = start_index , no = 0; no < length ; ++no, ++i){
     assert(curr_clusters[i].size() != 0);
     if(curr_clusters[i].size() > cluster_size*10){
       personal_clusters.push_back(curr_clusters[i]);
       continue;
     }
     vector<int> new_cluster;
     new_cluster.reserve(curr_clusters[i].size());
     if(curr_clusters[i].size() <= cluster_size_threshold){
       pers_wrongEle.reserve(pers_wrongEle.size() + curr_clusters[i].size());
       pers_wrongEle.insert(pers_wrongEle.end() , all(curr_clusters[i]));
       continue;
     }
    for(auto target_strand : curr_clusters[i]){
      int wrongNum = 0;
      for(auto other_strand : curr_clusters[i]){
        int dist = compute_distance(target_strand , other_strand , "edit");
        if(dist > fp_strand_edit_dis_threshold)
          ++wrongNum;
      }
      if(wrongNum > fp_strand_fraction_threshold*curr_clusters[i].size()){
        pers_wrongEle.push_back(target_strand);
      }
      else{
        new_cluster.push_back(target_strand);
      }
     }
     if(new_cluster.size() <= cluster_size_threshold){
       pers_wrongEle.reserve(pers_wrongEle.size() + curr_clusters[i].size()); 
       pers_wrongEle.insert(pers_wrongEle.end() , all(curr_clusters[i]));
       continue;
     }  
     personal_clusters.push_back(new_cluster);
  } 
  pthread_mutex_lock(&mutex1);
  clean_temp_clusters.insert(clean_temp_clusters.end(),all(personal_clusters));
  wrongEle.insert(wrongEle.end() , all(pers_wrongEle));
  pthread_mutex_unlock(&mutex1);  
  pthread_exit(NULL);
}

pair<vector<vector<int> >,vector<int> > clean_clusters(const vector<vector<int> >&clusters){
  clean_temp_clusters.clear();
  wrongEle.clear();
  if(clusters.size() == 0){
    return make_pair(clusters , wrongEle);
  }
  clean_temp_clusters.reserve(clusters.size());
  curr_clusters = clusters;
  int num_thread = min(max_num_thread,(int)curr_clusters.size());
  int each = curr_clusters.size()/num_thread;
  int extra = curr_clusters.size()%num_thread;
  pair<int,int> info_thread[num_thread];
  pthread_t my_thread[num_thread];
  pthread_mutex_init(&mutex1,NULL);
  int curr = 0;
  for(int i = 0;i < num_thread; ++i){
    info_thread[i] = make_pair(curr , each+(i<extra));
    curr += each+(i<extra);
    int res = pthread_create(&my_thread[i],NULL,cluster_cleaner,&info_thread[i]);
    if(res != 0){
      cout << "Error: p_thread create failed" << endl;
      exit(EXIT_FAILURE);
    }
  }
  for(int i = 0;i < num_thread ; ++i){
    pthread_join(my_thread[i],NULL);
  }
  return make_pair(clean_temp_clusters,wrongEle);
}

vector<vector<int> > recluster(const vector<vector<int> > &inp_clusters , const vector<int> &wrongEle){
  vector<vector<int> > wrong_initial_clusters;
  vector<vector<int> > clusters = inp_clusters;
  for(auto strand : wrongEle){
    pair<int, int> minId = make_pair(strand_length , -1); 
    int i = 0;
    for(auto cluster : clusters){
      assert(cluster.size() != 0);
      if(cluster.size() < 2*cluster_size){    //HARISH 
        int representative = cluster[random_no()%cluster.size()]; 
        //int prefix_edit_dist = compute_distance(strand.substr(0,remerging_compared_prefix_length),representative.substr(0,remerging_compared_prefix_length),"edit");
        //if(prefix_edit_dist <= remerging_compared_prefix_edit_dis_threshold)
          int edit_dist = compute_distance(strand , representative , "edit");
          minId = min(minId , make_pair(edit_dist , i)); 
      }
      ++i;
    }
    if(minId.first < r*remerging_edit_dis_threshold){
      assert(minId.second != -1);
      clusters[minId.second].push_back(strand);
    }
    else{
      vector<int> wrongCluster;
      wrongCluster.push_back(strand);
      wrong_initial_clusters.push_back(wrongCluster);
    }
  }
  if(wrong_initial_clusters.size() > 0){
    vector<vector<int> > wrongClusters = clustering(wrong_initial_clusters);  
    clusters.reserve(clusters.size() + wrongClusters.size()); 
    clusters.insert(clusters.end() ,all(wrongClusters));
  }
  return clusters;
}

struct thread_args{
  int start_index;
  int length;
  vector<vector<string> > *clusters;
  vector<vector<string> > *underlying_clusters;
};

void  *accuracy_parallel(void *args){
  thread_args info= *(static_cast<thread_args *>(args)); 
  int start_index = info.start_index , length = info.length;
  vector<vector<string> > &clusters = *(info.clusters);
  vector<vector<string> > &underlying_clusters = *(info.underlying_clusters);
  int *cnt = (int *)malloc(sizeof(int));
  (*cnt) = 0;
  for(int i = start_index,no = 0; no < length ; ++i,++no){
    unordered_set<string> checked_strand;
    gp_hash_table<int, int> freq;
    for(auto strand : underlying_clusters[i]){
      if(checked_strand.find(strand) != checked_strand.end())
        continue;
      checked_strand.insert(strand);
      for(auto indexes : fnc[strand])
        ++freq[indexes];
    } 
    for(auto ele : freq){
         if(clusters[ele.first].size() == ele.second && clusters[ele.first].size() >= gamma*underlying_clusters[i].size()){
            (*cnt) = (*cnt)+1;
            break;
         }
      } 
  }
  pthread_exit(cnt);
}

float accuracy(vector<vector<string> >&clusters , vector<vector<string> >&underlying_clusters){
  if(underlying_clusters.size() == 0){
    cout << "Underlying clusters are empty" << endl;
    return 0.0;
  }
  int num_thread = min(max_num_thread , (int)underlying_clusters.size());
  if(num_thread == 0){
    cout << "ERROR: max_num_thread set to zero" << endl;
    exit(EXIT_FAILURE);
  }
  int each = underlying_clusters.size()/num_thread;
  int extra = underlying_clusters.size()%num_thread;
  fnc.clear();
  pthread_t my_thread[num_thread];
  thread_args info_thread[num_thread];
  for(int i = 0, len = clusters.size() ; i < len ; ++i){
    for(auto strand: clusters[i])
      fnc[strand].push_back(i);
  }
  int curr = 0;
  for(int i = 0; i < num_thread ; ++i){
    info_thread[i].start_index = curr;
    info_thread[i].length = each+(i<extra);
    info_thread[i].clusters = &clusters;
    info_thread[i].underlying_clusters = &underlying_clusters;
    curr += each + (i < extra);
    int res = pthread_create(&my_thread[i],NULL,accuracy_parallel,&info_thread[i]);
    if(res != 0){
      cout << "Error: p_thread create failed" << endl;
      exit(EXIT_FAILURE);
    }
  }
  int cnt = 0;
  for(int i = 0;i < num_thread ; ++i){
    int *no;
    pthread_join(my_thread[i], (void **)&(no) );
    cnt += (*no);
    free(no);
  }
  return 1.0*cnt/underlying_clusters.size();
}


void write_to_file(ofstream &file , vector<vector<string> >&clusters){
  file << clusters.size() << endl;
  for(int i = 0, rows = clusters.size(); i < rows ; ++i){
    file << clusters[i].size() << '\n';
    for(int j = 0, cols = clusters[i].size(); j < cols ; ++j){
      file <<  clusters[i][j] << '\n';
    }
  }
}
void convert_to_string(vector<vector<int> >&clusters,vector<vector<string> >&v){
  v.resize(clusters.size());
  for(int i = 0,row = clusters.size(); i < row ; ++i){
    v[i].resize(clusters[i].size());
    for(int j = 0;j < clusters[i].size() ; ++j){
      v[i][j] = pool[clusters[i][j]];
    }
  }
}

gp_hash_table<int,int> edit_dis_freq;
gp_hash_table<int,int> q_gram_freq;
int mn_edit_dis;
void *parallel_preprocessing(void *args){
  vector<int> temp, temp1;
  temp.reserve(2*total_num_of_strands);
  bool ok1 = true , ok2 = true;
  for(int j = 0;j < 1 ; ++j){
    int idx = rand()%total_num_of_strands;
    int strange = 0;
    for(int i = 0;i < total_num_of_strands ; ++i){
      if(i == idx) continue;
      temp.push_back(compute_distance(idx , i , "edit"));
      temp1.push_back(compute_distance(idx , i , "hamming"));
      if(temp.back() < 0.4*strand_length) ++strange;
      assert(temp.back() < 2*strand_length);
    }
    if(strange > 3*cluster_size){
      if(j == 0) ok1 = false;
      if(j == 1) ok2 = false;
    }
  }
  pthread_mutex_lock(&mutex1);
 // assert(temp.size() = 2*(total_num_of_strands-1));
  for(int i = 0 , len = temp.size(); i < len ; ++i){
    if(!ok1 && i < total_num_of_strands-1) continue;
    if(!ok2 && i >= total_num_of_strands-1) continue;
    ++edit_dis_freq[temp[i]];
    ++q_gram_freq[temp1[i]];
  }
  pthread_mutex_unlock(&mutex1);
  pthread_exit(NULL);
}
void  preprocessing(){
  long long tot_len = 0;
  for(int i = 0;i < total_num_of_strands ; ++i)
    tot_len += pool[i].size();
  strand_length = round(tot_len*1.0/total_num_of_strands);
  cout << "Strand length: " << strand_length << endl;
  if(max_num_thread == 0){
    cout << "ERROR: max_num_thread set to zero"  << endl;
    exit(EXIT_FAILURE);
  }
  pthread_t my_thread[max_num_thread];
  pthread_mutex_init(&mutex1,NULL);
  for(int i = 0;i < max_num_thread ; ++i){
    int res = pthread_create(&my_thread[i],NULL,parallel_preprocessing,NULL);
    if(res != 0){
      cout << "Error: p_thread create failed" << endl;
      exit(EXIT_FAILURE);
    }
  }
  for(int i = 0;i < max_num_thread ; ++i){
    pthread_join(my_thread[i],NULL);
  }
  w = 0;
  long long pro = 1;
  while(pro*4ll < strand_length){
    pro *= 4ll;
    ++w;
  }
  ++w;
  cout << "Value of w: " << w << endl;
  pro = 1;
  l = 0;
  while(pro*4ll < total_num_of_strands/strand_length){
    pro *= 4ll;
    ++l;
  }
  ++l;
  l = max(3 , l-w);
  cout << "Value of l: " << l << endl;

  if(remove_false_positive_and_small_clusters){
    cluster_size_threshold = cluster_size/5;
    cout << "Value of cluster_size_threshold: " << cluster_size_threshold << endl;
    fp_strand_edit_dis_threshold = 1.1*r;
    cout << "Value of fp_strand_edit_dis_threshold: " << fp_strand_edit_dis_threshold << endl;
    fp_strand_fraction_threshold = 0.5;
    cout << "Value of fp_strand_fraction_threshold: " << fp_strand_fraction_threshold << endl;
  }
  if(remerge_removed_false_positives){
    cout << "Value of remerging_edit_dis_threshold: " << remerging_edit_dis_threshold << endl;
    remerging_edit_dis_threshold = 0.9;
  }  
  for(auto it : edit_dis_freq)
    debug << it.first << ' ' << it.second << endl;

  float edit_distribution[2*strand_length];
  float hamming_distribution[2*strand_length];
  for(int i = 0;i < 2*strand_length ; ++i){
    int sum = 0;
    int h_sum = 0;
    for(int j = -2 ; j <= 2 ; ++j){
      sum += edit_dis_freq[i+j];      
      h_sum += q_gram_freq[i+j];
    }
    edit_distribution[i] = sum*1.0/5;
    hamming_distribution[i] = h_sum*1.0/5;
  }
  int num_peak = 0;
  int num_h_peak = 0;
  int h_idx1 = -1 , h_idx2 = -1;
  int idx1 = -1 , idx2 = -1;
  int hamming_peak = 0; 
  for(int i = 0;i < 2*strand_length ; ++i){
    //if(i == 0 && edit_distribution[i] > 0.5+edit_distribution[i+1]) assert(false);
    //if(i == 0 && hamming_distribution[i] > 0.5+hamming_distribution[i+1]) assert(false);
    //if(i == 2*strand_length-1 && edit_distribution[i] > 0.5+edit_distribution[i-1]) assert(false);
    //if(i == 2*strand_length-1 && hamming_distribution[i] > 0.5+hamming_distribution[i-1]) assert(false);
    if(edit_distribution[i] > 0.5+edit_distribution[i-1] && edit_distribution[i] > 0.5+edit_distribution[i+1]){
      ++num_peak;
      if(num_peak == 1) idx1 = i;
      if(num_peak == 2) idx2 = i;
    }
   if(hamming_distribution[i] > 0.5+hamming_distribution[i-1] && hamming_distribution[i] > 0.5+hamming_distribution[i+1]){
    ++num_h_peak;
    if(num_h_peak == 1) h_idx1 = i;
    if(num_h_peak == 2) h_idx2 = i;
  }   
    if(i >= 1 && i < 2*strand_length-1){
      if(hamming_distribution[i] > hamming_distribution[hamming_peak])
        hamming_peak = i;
    }
  } 
  if(num_peak < 2 || edit_distribution[idx1] > (100.0/num_underlying_strands)*edit_distribution[idx2]){
    cout << "Not able to auto-tune r,theta_low,theta_high" << endl;
    exit(EXIT_FAILURE);
  }
  float mn = INF;
  for(int i = idx1 ; i <= idx2 ; ++i)
    mn = min(mn, edit_distribution[i]);

  int lft_idx = -1 , rht_idx = -1;
  for(int i = idx1 ;i <= idx2 ; ++i){
    if(edit_distribution[i] <= mn+1){
     lft_idx = i; 
     break;
    }
  }
  assert(lft_idx != -1);
  for(int i = idx2 ;i >= idx1 ; --i){
    if(edit_distribution[i] <= mn+1){
      rht_idx = i;
      break;
    }
  }
  assert(rht_idx != -1);
  r = (2*lft_idx + rht_idx)/3;
  cout << "Value of r: " << r << endl;
  if(total_num_of_strands > 1000){
     for(int i = hamming_peak ; i >= 0 ; --i){
       if(hamming_distribution[i]*2 < hamming_distribution[hamming_peak]){
         theta_high = i;
         break;
       }
     }
     if(h_idx2-h_idx1 > 0.2*strand_length){
       theta_low = h_idx1;
     }
     else{
       theta_low = 0.4*r;
     }
   }
  else{
    theta_low = -1;
    theta_high = INF;
  }  
  cout << "Value of theta_low: " << theta_low << endl;
  cout << "Value of theta_high: " << theta_high << endl;
}

void initalise();
int main(){
  ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
  cout << fixed << setprecision(4);
  ConfigParser("clustering_config_w.cfg");  
  initalise();
  debug.open(getValue("file_locations/debug_info"));
  max_num_thread = std::thread::hardware_concurrency();
  if(max_num_thread == 0) ++max_num_thread;
  srand(time(NULL));
  seed = time(NULL);
  pcg32_init(seed);
  time_t start_time; 
  time(&start_time);
  ifstream input_strands;
  input_strands.open(getValue("file_locations/input_pool"));
  string strand; 
  int idx = 0;
  vector<vector<int> > initial_clusters;
  while(input_strands >> strand){
    pool.push_back(strand);
    vector<int> initial_cluster;
    initial_cluster.push_back(idx);
    initial_clusters.push_back(initial_cluster);
    ++idx;
  }
  input_strands.close();
  if(initial_clusters.size() == 0){
    cout << "No input to cluster" << endl;
    exit(EXIT_FAILURE);
  }
  q_gram_seq_generator(gram_size , ""); 
  total_num_of_strands = pool.size(); 

  assert(total_num_of_strands == idx);
  q_gram_dict.resize(total_num_of_strands);
  cout << q_gram_dict.size() << endl;
  create_q_gram_dict();
  time_t q_gram_init_time; 
  time(&q_gram_init_time);
  if(print_time_stamps)
    cout << "--- " << q_gram_init_time-start_time  << " seconds for taking input strands and pre-computing q_grams---" << endl;

  if(auto_tuning){
    time_t pre_start , pre_end;
    time(&pre_start);
    preprocessing();
    time(&pre_end);
    cout << "--- " << pre_end-pre_start << " seconds for figuring out the optimal parameters---" << endl;
  }

  if(info_uc){
    time_t ul_s_time , ul_e_time;
    time(&ul_s_time);
    ifstream underlying_strands;
    underlying_strands.open(getValue("file_locations/underlying_clusters"));
    underlying_clusters.reserve(total_num_of_strands/cluster_size);
    string strand;
    getline(underlying_strands , strand);
    //underlying_strands >> strand;
    while(true){
      vector<string> cluster;
      cluster.reserve(2*cluster_size);
      bool last_cluster = 1;
      while(getline(underlying_strands , strand)){
        if(tolower(strand[1]) == 'l'){
          last_cluster = 0;
          break;
        }
        cluster.push_back(strand);
      }
      underlying_clusters.push_back(cluster);
      if(last_cluster) break;
    }
    cout << underlying_clusters.size() << endl;
    underlying_strands.close();
    if(underlying_clusters.size() == 0){
      cout << "Underlying clusters are absent" << endl;
      exit(EXIT_FAILURE);
    }
    time(&ul_e_time);
    if(print_time_stamps)
      cout << "--- " << ul_e_time-ul_s_time << " second for reading the underlying clusters---" << endl;
  }

  time_t  clus_time;
  time(&clus_time);
  temp_clusters.reserve(total_num_of_strands);

  vector<vector<int> > output_clusters = clustering(initial_clusters); 
  ofstream output_file;
  output_file.open(getValue("file_locations/output_clusters"));
  vector<vector<string> > paper_clusters;
  convert_to_string(output_clusters, paper_clusters);
  write_to_file(output_file,paper_clusters);
  output_file.close();
  if(print_time_stamps){
    cout << "Paper clusters:" << paper_clusters.size() << endl; 
    if(info_uc)
    cout << "Underlying clusters:" << underlying_clusters.size() << endl;
  }
  time_t clustering_time;
  time(&clustering_time);
  if(print_time_stamps)
    cout << "--- " << clustering_time-clus_time << " seconds for creating output clusters" << endl;
  
  if(remove_false_positive_and_small_clusters){
    pair<vector<vector<int> >,vector<int> > cleanup_output = clean_clusters(output_clusters);
    convert_to_string(cleanup_output.first , paper_clusters);
    ofstream kickWrongFile;
    kickWrongFile.open(getValue("file_locations/cleaned_up_clusters")); 
    write_to_file(kickWrongFile,paper_clusters);
    kickWrongFile.close();
    time_t clean_up_time;
    time(&clean_up_time);
    
    if(print_time_stamps){
      cout << "Clusters after cleanup:" << cleanup_output.first.size() << endl;
      cout << "Number of strands removed:" << cleanup_output.second.size() << endl;
      cout << "--- " << clean_up_time-clustering_time << " seconds for cleanup of output clusters" << endl;
    }

    if(remerge_removed_false_positives){
      vector<vector<int> > recycled_clusters = recluster(cleanup_output.first , cleanup_output.second); 
      convert_to_string(recycled_clusters, paper_clusters);
      ofstream recycle_file;
      recycle_file.open(getValue("file_locations/recycled_clusters"));

      write_to_file(recycle_file,paper_clusters);
      recycle_file.close();
      time_t recycle_time;
      time(&recycle_time);
      if(print_time_stamps){
        cout << "Clusters after remerging:" << recycled_clusters.size() << endl;
        cout << "--- " << recycle_time-clean_up_time << " seconds to recycle the output clusters" << endl;
      }
    }
  }
  time_t curr_time;
  time(&curr_time);
  
  if(info_uc)
    cout << "Accuracy (gamma=" << gamma << ") :" <<  accuracy(paper_clusters, underlying_clusters) << endl;

  time_t accuracy_time;
  time(&accuracy_time);

  if(print_time_stamps && info_uc)
    cout << "--- " << accuracy_time-curr_time << " seconds for computing accuracy" << endl;
  cout << "--- " << accuracy_time-start_time << " seconds for complete program execution" << endl; 
  //initial_clusters have the intial clustering
  return 0;
}
void initalise(){
  iterable.push_back('A');
  iterable.push_back('G');
  iterable.push_back('C');
  iterable.push_back('T');
  strand_length = stoi(getValue("parameters/strand_length")); 
  num_underlying_strands = stoi(getValue("parameters/num_underlying_strands"));
  remerging_edit_dis_threshold = stof(getValue("parameters/remerging_edit_dis_threshold"));  
  w = stoi(getValue("parameters/anchor_length"));
  l = stoi(getValue("parameters/hash_minus_anchor_length"));
  r = stoi(getValue("parameters/cluster_diameter"));
  cluster_size = stoi(getValue("parameters/cluster_size"));
  num_local_iter = stoi(getValue("parameters/num_local_iter"));
  num_global_iter = stoi(getValue("parameters/num_global_iter"));
  theta_low = stoi(getValue("parameters/theta_low"));
  theta_high = stoi(getValue("parameters/theta_high"));
  fp_strand_fraction_threshold = stof(getValue("parameters/fp_strand_fraction_threshold"));
  fp_strand_edit_dis_threshold = stoi(getValue("parameters/fp_strand_edit_dis_threshold"));
  cluster_size_threshold = stoi(getValue("parameters/cluster_size_threshold"));
  remerging_compared_prefix_edit_dis_threshold = stoi(getValue("parameters/remerging_compared_prefix_edit_dis_threshold"));
  remerging_compared_prefix_length = stoi(getValue("parameters/remerging_compared_prefix_length"));
  gamma = stof(getValue("parameters/gamma"));
  local_num_anchors = stoi(getValue("DEFAULT/local_num_anchors"));
  local_num_anchor_lists = stoi(getValue("DEFAULT/local_num_anchor_lists"));
  global_num_anchors = stoi(getValue("DEFAULT/global_num_anchors"));
  global_num_anchor_lists = stoi(getValue("DEFAULT/global_num_anchor_lists"));
  freq_representative = stoi(getValue("DEFAULT/freq_representative"));
  sorting_or_pairwise = stoi(getValue("flags/sorting_or_pairwise"));
  first_iteration_using_index = stoi(getValue("flags/first_iteration_using_index"));
  print_global_partition = stoi(getValue("flags/print_global_partition"));
  print_global_iterations = stoi(getValue("flags/print_global_iterations"));
  print_time_stamps = stoi(getValue("flags/print_time_stamps"));
  remove_false_positive_and_small_clusters = stoi(getValue("flags/remove_false_positive_and_small_clusters"));
  remerge_removed_false_positives = stoi(getValue("flags/remerge_removed_false_positives"));
  auto_tuning = stoi(getValue("flags/auto_tuning"));
  info_uc = stoi(getValue("flags/info_uc"));
}
