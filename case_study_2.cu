

// function definition
// "float* " means point to something.
void VecAdd(int N, float* A, float* B, float* C){
    for(int i=0; i<N; i++)
        C[i] = A[i] + B[i];

}

// ============================================================
// Make it become parallel function


// [Simple way to synchronize it] 1 block, N threads.
// Each thread process 1 addition. So the elements will be
// indexed as threadIdx.x

__global__ void VecAdd(float* A, float* B, float* C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}



int main(){

    // kernel invocation with N threads
    VecAdd<<<1,N>>>(Ah, Bh, Ch);

}


// ============================================================
// [General way to synchronize it] 
// Access 的順序會影響 memory access 的 pattern，讀資料的效能就有影響。
// index = blockIdx.x * blockDim.x + threadIdx.x

// Assume BS is block size. (num_threads_per_block)

__global__ void Add(int *a, int *b, int *c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}




int main(){
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    ...
    // kernel invocation with N threads
    Add<<<N/BS, BS>>>(d_a, d_b, d_c);
    ...
}


// ============================================================
// 如果 N/BS 沒辦法整除呢...取 ceiling。但是 mem access會發生一些問題。
// 例如：


__global__ void Add(int *a, int *b, int *c, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[i] = a[i] + b[i];
}


int main(void){
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    ...
    // kernel invocation with N threads
    Add<<<(N+BS-1)/BS, BS>>>(d_a, d_b, d_c, N);
    ...
}

// 事實上，GPU programming 會盡量避免 branch 生成，例如 if A else B
// 因為會浪費掉計算的 cycle，因為實際上會大家都做 A, Ｂ，再看誰不需要 A，把它丟掉。

// So... 最好還是 assure 會整除啦，就沒事了！
