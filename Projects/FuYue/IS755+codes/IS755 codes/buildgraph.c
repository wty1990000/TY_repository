#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <time.h>
#include <string>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <ctime>


using namespace std;

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


int gethour(int timestamp)
{
    char       buf[80];
    time_t rawtime=(time_t)timestamp;
    struct tm  ts;
    ts=*localtime(&rawtime);
//    strftime(buf, sizeof(buf), "%a %Y-%m-%d %H:%M:%S %Z", &ts);
//    printf("%s <%d>\n", buf, ts.tm_hour);
    return ts.tm_hour; 
}


// KL divergence
double KL(double *P, double *Q, int dim)
{
   double kl=0.0;
   for (int i=0; i<dim; i++)
   {
        if (P[i]>0.0 && Q[i]>0.0)
        {
                kl = kl + P[i]*log(P[i]/Q[i]);
        }
   }
   return kl;
}

void readLabels(int *true_labels, int dim)
{
   int i, ID, cat;
   for (i=0; i<dim; i++)
        true_labels[i]=-1;
   FILE *fp = fopen("venue_label.csv","r");
   while (fscanf(fp,"%d,%d", &ID, &cat)!=EOF){
//	printf("%d %d\n", ID, cat);
        true_labels[ID]=cat;
   }
   fclose(fp);

}


// a graph built from similarities. 2 venues are connected if they have high similiarities
void simGraph(double **chkp, int V, double *sumChkin)
{
  int v1, v2;
  double threshold = 0.8;
  double chkin_threshold = 1.0; //ignore venues with too few checkins
  FILE *fout=fopen("simgraph","w");
  for (v1=0; v1<V; v1++){
	if (sumChkin[v1]>=chkin_threshold){
	  for (v2=v1+1; v2<V; v2++){

		if (sumChkin[v2]>=chkin_threshold){
			double p_wt = (KL(chkp[v1],chkp[v2],24)+KL(chkp[v2],chkp[v1],24))/2.0; //symmetric distance measure
			p_wt = exp(-p_wt); //convert to similarity
			if (p_wt>=threshold)
				fprintf(fout,"%d %d %lf\n", v1, v2, p_wt);
		}
	  }
	}
  }
  fclose(fout);
}



// convert edge graph from pure jaccard sim to one with weighted sum of jaccard and temporal chkin similarity
void augmentgraph(double **chkp, double lambda, double lambdaBar)
{
  // Iterate thru edge graph to update edge weight
  int v1, v2, larger=0, smaller=0;
  double wt;
  FILE *fp = fopen("ISC_01component","r");
  FILE *fout = fopen("augmented","w");
  while (fscanf(fp,"%d %d %lf", &v1, &v2, &wt)!=EOF){
	//new wt = lambda*old wt + (1-lambda)*KL-div
	double p_wt = (KL(chkp[v1],chkp[v2],24)+KL(chkp[v2],chkp[v1],24))/2.0; //symmetric distance measure
	p_wt = exp(-p_wt); //convert to similarity
	double new_wt = (lambdaBar*wt + lambda*p_wt)/2.0;
	if (new_wt>wt)
		larger++;
	else
		smaller++;
	fprintf(fout,"%d %d %lf\n", v1, v2, new_wt);
  }
  fclose(fp);
  fclose(fout);
  printf("larger %d   smaller %d\n", larger, smaller);

}

// read venue node mapping
// bin their checkin.. convert into probability
// 
void buildgraph()
{
  int i,j,id, V;
  double lambda = 0.1, lambdaBar=1.0-lambda;
  vector<string> venues;
  map<string, int> indices;
  string line;
  ifstream myfile_word ("venue_new_old.txt"); //read mapping
  if (myfile_word.is_open())
  {
    while ( getline (myfile_word,line) )
    {
      std::vector<std::string> x = split(line,',');
      id = atoi(x[0].c_str());
      venues.push_back(x[1]);	
      indices[x[1]]=id;	
      //cout<<id<<","<<venue<<endl;      	
    }
    myfile_word.close();
  }
  V = (int)venues.size();
  
  int *true_labels=new int[V];
  readLabels(true_labels, V);
  double prior = 0.0; //1.0/24.0;
  double **chkp=new double*[V];
  for (i=0; i<V; i++){
	chkp[i]=new double[24]; 
	for (j=0; j<24; j++)
		chkp[i][j]=prior;
  }
  double **summary = new double*[9];  //9 categories
  for (i=0; i<9; i++){
	summary[i]=new double[24];
	for (j=0; j<24; j++)
		summary[i][j]=0.0;
  }
  //go thru checkin file to build a vector for each venue
  int cc=0;
  ifstream file_chkin ("checking_local_dedup.txt");
  if (file_chkin.is_open())
  {
    while ( getline (file_chkin,line) )
    {
      std::vector<std::string> x = split(line,','); //uid,vid,time
      if (indices.find(x[1])!=indices.end()){	
	// cout<<indices[x[1]]<<" "<<x[1]<<endl;
	int timestamp = atoi(x[2].c_str());
	int h=gethour(timestamp);
	int v = indices[x[1]];
	chkp[v][h] += 1.0;
	if (true_labels[v]!=-1){
		//printf("%d %d\n", true_labels[v], v);
		summary[true_labels[v]][h] += 1.0;
	}
      }
    }
    file_chkin.close();
  }
  // print checkin freq binned by hour for different categories
  FILE *fp = fopen("summary.txt","w");
  for (i=0; i<9; i++){
	for (j=0; j<24; j++)
		fprintf(fp, "%lf ", summary[i][j]);
  	fprintf(fp,"\n"); 
  }
  fclose(fp);
  double *sumChkin = new double[V]; //store count of chkin per venue
  // convert to probabilities
  for (i=0; i<V; i++){
	double sum = 0.0;
	for (j=0; j<24; j++)
		sum += chkp[i][j];
	sumChkin[i]=sum;
/*
	if (i<=2){
		printf("%lf\n", sum);
		for (j=0; j<24; j++)
			printf("%lf ", chkp[i][j]);
		printf("\n");
	}
*/
	// some venues are in IS755_checkin_small.txt but not in checking_local_dedup.txt
	for (j=0; j<24; j++){
		if (sum != 0.0)
			chkp[i][j]=chkp[i][j]/sum;
	}
  }

  augmentgraph(chkp, lambda, lambdaBar);
  //simGraph(chkp, V, sumChkin);

  for (i=0; i<V; i++)
  	delete[] chkp[i];
  delete[] chkp;
  for (i=0; i<9; i++)
	delete[] summary[i];
  delete[] summary;
  delete[] true_labels;
  delete[] sumChkin;
}

void example()
{
    time_t     now;
    struct tm  ts;
    char       buf[80];
    // Get current time
    time(&now);
    // Format time, "ddd yyyy-mm-dd hh:mm:ss zzz"
    //ts = *localtime(&now);
    time_t rawtime=1351556855;
    ts=*localtime(&rawtime);
    strftime(buf, sizeof(buf), "%a %Y-%m-%d %H:%M:%S %Z", &ts);
    printf("%s\n", buf);
    printf("%d\n", ts.tm_hour);
}

int main(void)
{
    buildgraph();
    return 0;
}
