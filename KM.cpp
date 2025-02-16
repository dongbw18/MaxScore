#include <stdio.h>
#include <string.h>
#include <fstream>
#include <algorithm>
using namespace std;

const int N = 512, E = 16, topK = 2, capacity = N * topK / E;
const float INF = 1e9;

extern "C" {

    void Dfs(int rt, int* havx, int* havy, int* pre){
        if(rt == 0) return;
        havy[rt] = pre[rt];
        Dfs(havx[havy[rt]], havx, havy, pre);
        havx[havy[rt]] = rt;
    } 

    void Bfs(int S, float* score, float* lx, float* ly, int* j2expert, float* up, int* q, int* visx, int* visy, int tim, int* pre, int* havx, int* havy){
        for(int i = 0; i < N; i++) up[i + 1] = INF;
        int s, t; q[s=t=1] = S;
        while(true){
            while(s <= t){
                int rt = q[s++];
                visx[rt] = tim;
                for(int i = 1; i <= N; i++) if(visy[i] != tim){
                    float tmp = lx[rt] + ly[i] - score[(rt - 1) * E + j2expert[i - 1]];
                    if(tmp == 0){
                        visy[i] = tim, pre[i] = rt;
                        if(havy[i] == 0){ Dfs(i, havx, havy, pre); return;}
                        q[++t] = havy[i];
                    }
                    else { if(tmp < up[i]) up[i] = tmp, pre[i] = rt;}
                }   
            }
            float tmp = INF;
            for(int i = 1; i <= N; i++) if(visy[i] != tim) tmp = min(up[i], tmp);
            for(int i = 1; i <= N; i++) if(visx[i] == tim) lx[i] -= tmp;
            for(int i = 1; i <= N; i++) if(visy[i] == tim) ly[i] += tmp; else up[i] -= tmp;
            for(int i = 1; i <= N; i++) if(visy[i] != tim && up[i] == 0) {
                visy[i] = tim;
                if(havy[i] == 0) { Dfs(i, havx, havy, pre); return; }
                q[++t] = havy[i];
            }
        }
    }

    void KM(float* score, int* indices1, int* indices2){
        float* lx = new float[N + 1];
        float* ly = new float[N + 1];
        int* cnt_expert = new int[E];
        int* j2expert = new int[N];
        float* up = new float[N + 1];
        int *q = new int[N + 1];
        int *visx = new int[N + 1];
        int *visy = new int[N + 1];
        int tim = 0;
        int *pre = new int[N + 1];
        int *havx = new int[N + 1];
        int *havy = new int[N + 1];
        // float score_total = 0;

        for(int e = 0; e < E; e++) cnt_expert[e] = 0;
        for(int i = 0; i <= N; i++){
            lx[i] = ly[i] = 0;
            visx[i] = visy[i] = 0;
            pre[i] = 0;
            havx[i] = havy[i] = 0;
        }

        for(int i = 0; i < N; i++){
            indices1[i] = -1;
            for(int j = 0; j < E; j++) 
                if(cnt_expert[j] < capacity){
                    if(indices1[i] < 0 || score[i * E + j] > score[i * E + indices1[i]])
                        indices1[i] = j;
                }
            cnt_expert[indices1[i]] ++;
            // score_total += score[i * E + indices1[i]];
            score[i * E + indices1[i]] = 0;
            for(int j = 0; j < E; j++) 
                lx[i + 1] = max(lx[i + 1], score[i * E + j]);
        }

        for(int e = 0, i = 0; e < E; e ++)
            for(int j = 0; j + cnt_expert[e] < capacity; i++, j++)
                j2expert[i] = e;

        for(int i = 0; i < N; i++) Bfs(i + 1, score, lx, ly, j2expert, up, q, visx, visy, ++tim, pre, havx, havy);
        
        // for(int i = 0; i < N; i++) score_total += lx[i + 1] + ly[i + 1];
        // printf("CKM Max Flow: %d with Max Score %f\n", N * 2, score_total);
        for(int i = 0; i < N; i++) indices2[i] = j2expert[havx[i + 1] - 1];

        delete lx;
        delete ly;
        delete cnt_expert;
        delete j2expert;
        delete up;
        delete q;
        delete visx;
        delete visy;
        delete pre;
        // return score_total;
    }

}