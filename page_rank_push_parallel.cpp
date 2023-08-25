#include "core/graph.h"
#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include "pthread.h"

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
typedef double PageRankType;
#endif

pthread_mutex_t mymutex = PTHREAD_MUTEX_INITIALIZER;

#define NUM_THREADS 5000000
pthread_mutex_t mutexes[NUM_THREADS];

Graph* mygraph;
int MAXiter;
PageRankType *pr_curr;
PageRankType *pr_next;
double* timetakenarr;

class ThreadDetail {
    public:
    int ThreadID;
    int ThreadStartIndex;
    int ThreadEndIndex;
   double ThreadTimeTaken;
   CustomBarrier* Thread_barrier;
};

void* PRFunc(void* threadarr){
  ThreadDetail* threadinfo;
  threadinfo = (ThreadDetail*)threadarr;
  timer t1;
  double time_taken = 0.0;
  t1.start();
  int start = threadinfo->ThreadStartIndex;
  int end =  threadinfo->ThreadEndIndex;

  int computations = 0;
  int initer=0;
  int invloop =0;
  //std::cout<<"Thread - " << threadinfo->ThreadID <<" - start and end range: " << start<<"  -  "<<end<<"\n";

  for (int iter = 0; iter < MAXiter; iter++) {
    initer++;
    // for each vertex 'u', process all its outNeighbors 'v'
    for (uintV u = start; u <= end; u++) {
      invloop++;
      uintE out_degree = mygraph->vertices_[u].getOutDegree();
      for (uintE i = 0; i < out_degree; i++) {
        uintV v = mygraph->vertices_[u].getOutNeighbor(i);
        pthread_mutex_lock(&mutexes[v]);
        pr_next[v] += (pr_curr[u] / (PageRankType) out_degree);
        pthread_mutex_unlock(&mutexes[v]);

        computations++;
      }
    }
    threadinfo->Thread_barrier->wait();


    for (uintV v = start; v <= end; v++) {
      pr_next[v] = PAGE_RANK(pr_next[v]);

      // reset pr_curr for the next iteration
      pr_curr[v] = pr_next[v];
      pr_next[v] = 0.0;
      }
  
    threadinfo->Thread_barrier->wait();
    }

    time_taken=t1.stop();
    timetakenarr[threadinfo->ThreadID] = time_taken;
    std::cout<<"ThreadID: "<<threadinfo->ThreadID<<" For iter: " << initer<< " inVloop: "<< invloop<< " computations: " <<computations<<"\n";

  }

void pageRankParallel(Graph &g, int max_iters, int t_threads) {
  for (int i = 0; i < t_threads; i++){
        pthread_mutex_init(&mutexes[i], NULL);
    }


  timetakenarr = new double[t_threads+1];
  CustomBarrier my_barrier(t_threads);
  uintV n = g.n_;

  pr_curr = new PageRankType[n+1];
  pr_next = new PageRankType[n+1];

  for (uintV i = 0; i < n; i++) {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  // Pull based pagerank
  timer t1;
  double time_taken = 0.0;
  // Create threads and distribute the work across T threads

  pthread_t *pthread_arr;
  pthread_arr = new pthread_t[t_threads];
  ThreadDetail *DetailsArr;
  DetailsArr = new ThreadDetail[t_threads];
  mygraph = &g;
  MAXiter = max_iters;

  /*
    int ThreadID;
    int ThreadStartIndex;
    int ThreadEndIndex;
    int ThreadTimeTaken;
  */
  // -------------------------------------------------------------------
    int VerticesPerThread = n/t_threads;
    int VerticesLastThread = n%t_threads + VerticesPerThread;


    t1.start();
  for(int i = 0; i < t_threads; i++){
    DetailsArr[i].Thread_barrier=&my_barrier;
    DetailsArr[i].ThreadID = i;
    DetailsArr[i].ThreadStartIndex = (i*VerticesPerThread);
    DetailsArr[i].ThreadEndIndex = (i*VerticesPerThread)+VerticesPerThread-1;
    if(i==(t_threads-1)){
        DetailsArr[i].ThreadEndIndex = n;
    }
    //std::cout<<"Thread "<<i<<" Start: "<<DetailsArr[i].ThreadStartIndex<< " Ends: "<<DetailsArr[i].ThreadEndIndex<<"\n";
    pthread_create(&pthread_arr[i], NULL, PRFunc, (void*)&DetailsArr[i]);
    }
    void* retval = 0;
  for(int i = 0 ; i < t_threads; i++){
      pthread_join(pthread_arr[i], &retval);
    }
/////////////
  time_taken = t1.stop();
  // -------------------------------------------------------------------
  std::cout << "thread_id, time_taken" << std::endl;
  for(int i =0; i < t_threads; i++){
    std::cout << i <<", " << timetakenarr[i]<< std::endl; 
  }


  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++) {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page ranks : " << sum_of_page_ranks << "\n";
  std::cout << "Time taken (in seconds) : " << time_taken << "\n";
  ///delete[] pr_curr;
  //delete[] pr_next;
}


int main(int argc, char *argv[]) {
  cxxopts::Options options(
      "page_rank_push",
      "Calculate page_rank using serial and parallel execution");
  options.add_options(
      "",
      {
          {"nThreads", "Number of Threads",
           cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_THREADS)},
          {"nIterations", "Maximum number of iterations",
           cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
          {"inputFile", "Input graph file path",
           cxxopts::value<std::string>()->default_value(
               "/scratch/input_graphs/roadNet-CA")},
      });

  auto cl_options = options.parse(argc, argv);
  uint n_threads = cl_options["nThreads"].as<uint>();
  uint max_iterations = cl_options["nIterations"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();

#ifdef USE_INT
  std::cout << "Using INT" << std::endl;
#else
  std::cout << "Using DOUBLE" << std::endl;
#endif
  std::cout << std::fixed;
  std::cout << "Number of Threads : " << n_threads << std::endl;
  std::cout << "Number of Iterations: " << max_iterations << std::endl;

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";
  pageRankParallel(g, max_iterations, n_threads);

  return 0;
}
