#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

typedef long long ll;

int id(int x, int y, int z, int a, int b, int c) {
    return c * (x + y*a + z*a*b);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    double t1, t2, t3;
    t1 = MPI_Wtime();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check for the proper number of input arguments.
    if (argc < 10) {
        if (rank == 0)
            printf("Usage: %s data_file PX PY PZ NX NY NZ NC output_file\n", argv[0]);
        MPI_Finalize();
        return -1;
    }
    
    // Input arguments
    char *data_file = argv[1];
    int PX = atoi(argv[2]);
    int PY = atoi(argv[3]);
    int PZ = atoi(argv[4]);
    int NX = atoi(argv[5]);
    int NY = atoi(argv[6]);
    int NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    char *output_file = argv[9];

    // Verify that the number of processes matches the 3D process grid.
    if (PX * PY * PZ != size) {
        if (rank == 0)
            printf("Error: Number of processes must equal PX * PY * PZ\n");
        MPI_Finalize();
        return -1;
    }

//----- Parallel Read from same file ---------    

    MPI_File file;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, data_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    ll N = NX*NY*NZ*NC;
    ll sendcount = N/size;
    MPI_Offset offset = rank*sendcount* sizeof(double);

    double *sendbuf = (double *)malloc(sendcount*sizeof(double));

    MPI_File_read_at_all(file, offset, sendbuf, sendcount, MPI_DOUBLE, &status);

    MPI_File_close(&file);

    double *recvbuf = NULL;
    if (rank == 0) {
        recvbuf = (double *)malloc(sizeof(double) * N);
    }

    // Gathering all data at rank 0 to redistribute
    MPI_Gather(sendbuf, sendcount, MPI_DOUBLE, recvbuf, sendcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(sendbuf);

//------ Distibuting data to all processes from rank 0 -------

    // Determine the sub-volume dimensions for each process.
    int sub_nx = NX / PX;
    int sub_ny = NY / PY;
    int sub_nz = NZ / PZ;
    ll local_data_size = sendcount;
    double *local_data = (double *)malloc(sizeof(double) * local_data_size);

    // Rearranging data so that every process's grid points are placed consecutively
    double *full_data = NULL;
    if (rank == 0) {
        full_data = (double *)malloc(sizeof(double) * N);
        int *curr = (int *)malloc(sizeof(int) * size); 
        for (int i=0; i<size; i++)
            curr[i] = 0;

        int ptr = 0;
        for (int z=0; z<NZ; z++) {
            for (int y=0; y<NY; y++) {
                for (int x=0; x<NX; x++) {
                    int r = x/sub_nx + (y/sub_ny)*PX + (z/sub_nz)*(PX*PY);
                    for (int i=0; i<NC; i++) {
                        full_data[r*local_data_size + curr[r]] = recvbuf[ptr++];
                        curr[r]++;
                    }
                }
            }
        }
        free(curr);
        free(recvbuf);
    }


    // Distribute the sub-volume data to each process.
    MPI_Scatter(full_data, local_data_size, MPI_DOUBLE,local_data, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        free(full_data);
    }
    t2 = MPI_Wtime();

//-----  Data sharing from neighbour ranks/subdomains -------

    int proc_x = rank%PX;
    int proc_y = (rank/PX)%PY;
    int proc_z = rank/(PX*PY);
    int neigh[6];
    neigh[0] = (proc_x > 0) ? rank-1 : MPI_PROC_NULL;
    neigh[1] = (proc_x < PX-1) ? rank+1 : MPI_PROC_NULL;
    neigh[2] = (proc_y > 0) ? rank-PX : MPI_PROC_NULL;
    neigh[3] = (proc_y < PY-1) ? rank+PX : MPI_PROC_NULL;
    neigh[4] = (proc_z > 0) ? rank-(PX*PY) : MPI_PROC_NULL;
    neigh[5] = (proc_z < PZ-1) ? rank+(PX*PY) : MPI_PROC_NULL;

    MPI_Request reqs[12];
    int req_count = 0;


// ----- Exchange X-direction (left and right faces) -----
    int size_x = sub_ny*sub_nz*NC;
    double *halo_left = (double *)malloc(size_x * sizeof(double));
    double *halo_right = (double *)malloc(size_x * sizeof(double));
    double *send_left  = (double *)malloc(size_x * sizeof(double));
    double *send_right = (double *)malloc(size_x * sizeof(double));

    int curr = 0;
    for (int z=0; z<sub_nz; z++) {
        for (int y=0; y<sub_ny; y++) {
            for (int nc=0; nc<NC; nc++) {
                send_left[curr++] = local_data[id(0,y,z,sub_nx,sub_ny,NC)+nc];
            }
        }
    }
    curr = 0;
    for (int z=0; z<sub_nz; z++) {
        for (int y=0; y<sub_ny; y++) {
            for (int nc=0; nc<NC; nc++) {
                send_right[curr++] = local_data[id(sub_nx-1,y,z,sub_nx,sub_ny,NC)+nc];
            }
        }
    }
    if (neigh[0] != MPI_PROC_NULL) {
        MPI_Irecv(halo_left, size_x, MPI_DOUBLE, neigh[0], 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(send_left, size_x, MPI_DOUBLE, neigh[0], 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (neigh[1] != MPI_PROC_NULL) {
        MPI_Irecv(halo_right, size_x, MPI_DOUBLE, neigh[1], 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(send_right, size_x, MPI_DOUBLE, neigh[1], 0, MPI_COMM_WORLD, &reqs[req_count++]);      
    }
    free(send_left);
    free(send_right);


// ----- Exchange Y-direction (down and up faces) -----
    int size_y = sub_nx*sub_nz*NC;
    double *halo_down = (double *)malloc(size_y * sizeof(double));
    double *halo_up = (double *)malloc(size_y * sizeof(double));
    double *send_down  = (double *)malloc(size_y * sizeof(double));
    double *send_up = (double *)malloc(size_y * sizeof(double));

    curr = 0;
    for (int z=0; z<sub_nz; z++) {
        for (int x=0; x<sub_nx; x++) {
            for (int nc=0; nc<NC; nc++) {
                send_down[curr++] = local_data[id(x,0,z,sub_nx,sub_ny,NC)+nc];
            }
        }
    }
    curr = 0;
    for (int z=0; z<sub_nz; z++) {
        for (int x=0; x<sub_nx; x++) {
            for (int nc=0; nc<NC; nc++) {
                send_up[curr++] = local_data[id(x,sub_ny-1,z,sub_nx,sub_ny,NC)+nc];
            }
        }
    }
    if (neigh[2] != MPI_PROC_NULL) {
        MPI_Irecv(halo_down, size_y, MPI_DOUBLE, neigh[2], 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(send_down, size_y, MPI_DOUBLE, neigh[2], 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (neigh[3] != MPI_PROC_NULL) {
        MPI_Irecv(halo_up, size_y, MPI_DOUBLE, neigh[3], 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(send_up, size_y, MPI_DOUBLE, neigh[3], 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    free(send_down);
    free(send_up);

// ----- Exchange Z-direction (back and front faces) -----
    int size_z = sub_nx*sub_ny*NC;
    double *halo_back  = (double *)malloc(size_z * sizeof(double));
    double *halo_front  = (double *)malloc(size_z * sizeof(double));
    double *send_back  = (double *)malloc(size_z * sizeof(double));
    double *send_front = (double *)malloc(size_z * sizeof(double));

    curr = 0;
    for (int y=0; y<sub_ny; y++) {
        for (int x=0; x<sub_nx; x++) {
            for (int nc=0; nc<NC; nc++) {
                send_back[curr++] = local_data[id(x,y,0,sub_nx,sub_ny,NC)+nc];
            }
        }
    }
    curr = 0;
    for (int y=0; y<sub_ny; y++) {
        for (int x=0; x<sub_nx; x++) {
            for (int nc=0; nc<NC; nc++) {
                send_front[curr++] = local_data[id(x,y,sub_nz-1,sub_nx,sub_ny,NC)+nc];
            }
        }
    }
    if (neigh[4] != MPI_PROC_NULL) {
        MPI_Irecv(halo_back, size_z, MPI_DOUBLE, neigh[4], 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(send_back, size_z, MPI_DOUBLE, neigh[4], 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (neigh[5] != MPI_PROC_NULL) {
        MPI_Irecv(halo_front, size_z, MPI_DOUBLE, neigh[5], 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(send_front, size_z, MPI_DOUBLE, neigh[5], 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    free(send_back);
    free(send_front);

    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

//------- Local and global extremas in subdomain ------   

    int *local_min = (int *)malloc(NC * sizeof(int));
    int *local_max = (int *)malloc(NC * sizeof(int));
    double *local_global_min = (double *)malloc(NC * sizeof(double));
    double *local_global_max = (double *)malloc(NC * sizeof(double));

    for (int i=0; i<NC; i++) {
        local_min[i] = local_max[i] = 0;
        local_global_min[i] = 1e9;
        local_global_max[i] = -1e9;
    }

    for (int z=0; z<sub_nz; z++) {
        for (int y=0; y<sub_ny; y++) {
            for (int x=0; x<sub_nx; x++) {
                for (int i=0; i<NC; i++) {
                    double val = local_data[id(x,y,z,sub_nx,sub_ny,NC)+i], val1;
                    if (val < local_global_min[i]) local_global_min[i] = val;
                    if (val > local_global_max[i]) local_global_max[i] = val;
                    int f1 = 1, f2 = 1;

                    if (x==0) {
                        if (neigh[0] != MPI_PROC_NULL) {
                            val1 = halo_left[(y+z*sub_ny)*NC + i];
                            if (val >= val1) f1 = 0;
                            if (val <= val1) f2 = 0;
                        }
                    }
                    else {
                        val1 = local_data[id(x-1,y,z,sub_nx,sub_ny,NC) + i];
                        if (val >= val1) f1 = 0;
                        if (val <= val1) f2 = 0;
                    }
                    if (x==sub_nx-1) {
                        if (neigh[1] != MPI_PROC_NULL) {
                            val1 = halo_right[(y+z*sub_ny)*NC + i];
                            if (val >= val1) f1 = 0;
                            if (val <= val1) f2 = 0;
                        }
                    }
                    else {
                        val1 = local_data[id(x+1,y,z,sub_nx,sub_ny,NC) + i];
                        if (val >= val1) f1 = 0;
                        if (val <= val1) f2 = 0;
                    }

                    if (y==0) {
                        if (neigh[2] != MPI_PROC_NULL) {
                            val1 = halo_down[(x+z*sub_nx)*NC + i];
                            if (val >= val1) f1 = 0;
                            if (val <= val1) f2 = 0;
                        }
                    }
                    else {
                        val1 = local_data[id(x,y-1,z,sub_nx,sub_ny,NC) + i];
                        if (val >= val1) f1 = 0;
                        if (val <= val1) f2 = 0;
                    }
                    if (y==sub_ny-1) {
                        if (neigh[3] != MPI_PROC_NULL) {
                            val1 = halo_up[(x+z*sub_nx)*NC + i];
                            if (val >= val1) f1 = 0;
                            if (val <= val1) f2 = 0;
                        }
                    }
                    else {
                        val1 = local_data[id(x,y+1,z,sub_nx,sub_ny,NC) + i];
                        if (val >= val1) f1 = 0;
                        if (val <= val1) f2 = 0;
                    }

                    if (z==0) {
                        if (neigh[4] != MPI_PROC_NULL) {
                            val1 = halo_back[(x+y*sub_nx)*NC + i];
                            if (val >= val1) f1 = 0;
                            if (val <= val1) f2 = 0;
                        }
                    }
                    else {
                        val1 = local_data[id(x,y,z-1,sub_nx,sub_ny,NC) + i];
                        if (val >= val1) f1 = 0;
                        if (val <= val1) f2 = 0;
                    }
                    if (z==sub_nz-1) {
                        if (neigh[5] != MPI_PROC_NULL) {
                            val1 = halo_front[(x+y*sub_nx)*NC + i];
                            if (val >= val1) f1 = 0;
                            if (val <= val1) f2 = 0;
                        }
                    }
                    else {
                        val1 = local_data[id(x,y,z+1,sub_nx,sub_ny,NC) + i];
                        if (val >= val1) f1 = 0;
                        if (val <= val1) f2 = 0;
                    }

                    local_min[i] += f1;
                    local_max[i] += f2;
                }
            }
        }
    }
    free(local_data);
    free(halo_back);
    free(halo_down);
    free(halo_front);
    free(halo_left);
    free(halo_right);
    free(halo_up);

//------ Collecting local extremas count and global extremas at rank 0 --------
    
    int *local_minimas = NULL, *local_maximas = NULL;
    double *global_min = NULL, *global_max = NULL;
    if (rank == 0) {
        local_minimas = (int *)malloc(sizeof(int) * NC);
        local_maximas = (int *)malloc(sizeof(int) * NC);
        global_min = (double *)malloc(sizeof(double) * NC);
        global_max = (double *)malloc(sizeof(double) * NC);
    }
    for (int i=0; i<NC; i++) {
        MPI_Reduce(&local_min[i], &local_minimas[i], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max[i], &local_maximas[i], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_global_min[i], &global_min[i], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_global_max[i], &global_max[i], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    free(local_min);
    free(local_max);
    free(local_global_max);
    free(local_global_min);

    t3 = MPI_Wtime();

//------Max time across all processes

    double read_time = t2-t1, main_time = t3-t2, total_time = t3-t1;
    double max_read_time, max_main_time, max_total_time;

    MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&main_time, &max_main_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

//------ Output results from rank 0 -----
    if (rank == 0) {
        FILE *fp = fopen(output_file, "a");
        if (fp == NULL) {
            printf("Error opening output file %s\n", output_file);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        for (int i=0; i<NC; i++) {
            fprintf(fp, "(%d, %d)%s", local_minimas[i], local_maximas[i], (i==NC-1) ? "\n" : ", ");
        }
        for (int i=0; i<NC; i++) {
            fprintf(fp, "(%.4f, %.4f)%s", global_min[i], global_max[i], (i==NC-1) ? "\n" : ", ");
        }
        fprintf(fp, "%f %f %f\n", max_read_time, max_main_time, max_total_time);
        fclose(fp);
        free(local_minimas);
        free(local_maximas);
        free(global_max);
        free(global_min);
    }

    MPI_Finalize();
    return 0;
}
