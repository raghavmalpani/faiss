/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/impl/HNSW.h>



double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void greedyError(faiss::UpperLevelSearchStat u, int x, int d, std::vector<float> database, faiss::IndexHNSWFlat index) {
    int i0 = x;
    int i1 = i0 + 1;
    int k = 5;
    int nq = 1;
    std::vector<faiss::Index::idx_t> nns(k * nq);
    std::vector<float> dis(k * nq);
    std::vector<float> queries;

    queries.resize(nq * d);
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < d; j++) {
            queries[(i - i0) * d + j] = database[i * d + j];
        }
    }

    index.search(nq, queries.data(), k, dis.data(), nns.data());
}

int main() {

    // int d = 100;
    // int maxM = 16;
    // int train = 1;
    // int nt = 1;
    // faiss::IndexHNSWFlat index(d,maxM);
    // index.verbose = true;
    // index.hnsw.efConstruction = 40;
    // index.metric_type = faiss::METRIC_INNER_PRODUCT;

    // std::mt19937 rng;

    // std::vector<float> trainvecs(nt * d);
    // std::uniform_real_distribution<> distrib;
    // for (size_t i = 0; i < nt * d; i++) {
    //     trainvecs[i] = distrib(rng);
    // }

    // printf("hi");

    // index.add(nt, trainvecs.data());

    // index.search(nq, queries.data(), k, dis.data(), nns.data());

    for (int i = 0; i < 1000; i++){
        try{
        
        
        
        double t0 = elapsed();

        // dimension of the vectors to index
        int d = 64;

        int maxM = 32;

        // size of the database we plan to index
        size_t nb = 20 * 100;

        // make a set of nt training vectors in the unit cube
        // (could be the database)
        size_t nt = 10 * 100;

        // make the index object and train it
        //faiss::IndexFlatL2 coarse_quantizer(d);

        // a reasonable number of centroids to index nb vectors
        int ncentroids = int(4 * sqrt(nb));

        // the coarse quantizer should not be dealloced before the index
        // 4 = nb of bytes per code (d must be a multiple of this)
        // 8 = nb of bits per sub-code (almost always 8)
        faiss::IndexHNSWFlat index(d, maxM);

        std::mt19937 rng;

        printf("ef = %d\n", index.hnsw.efSearch);
         printf(index.hnsw.search_bounded_queue ? "index.hnsw.search_bounded_queue: true\n" : "index.hnsw.search_bounded_queue: false\n");

        { // training
            printf("[%.3f s] Generating %ld vectors in %dD for training\n",
                elapsed() - t0,
                nt,
                d);

            std::vector<float> trainvecs(nt * d);
            std::uniform_real_distribution<> distrib;
            for (size_t i = 0; i < nt * d; i++) {
                trainvecs[i] = distrib(rng);
            }

            printf("[%.3f s] Training the index\n", elapsed() - t0);
            index.verbose = true;

            index.train(nt, trainvecs.data());
        }

        { // I/O demo
            const char* outfilename = "/tmp/index_trained.faissindex";
            printf("[%.3f s] storing the pre-trained index to %s\n",
                elapsed() - t0,
                outfilename);

            write_index(&index, outfilename);
        }

        size_t nq;
        std::vector<float> queries;
        std::vector<float> database(nb * d);

        { // populating the database
            printf("[%.3f s] Building a dataset of %ld vectors to index\n",
                elapsed() - t0,
                nb);

            
            std::uniform_real_distribution<> distrib;
            for (size_t i = 0; i < nb * d; i++) {
                database[i] = distrib(rng);
            }

            printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

            printf("%ld\n", nb);

            index.add(nb, database.data());

            // printf("[%.3f s] imbalance factor: %g\n",
            //        elapsed() - t0,
            //        index.invlists->imbalance_factor());

            // remember a few elements from the database as queries

            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, 19950); 

            int i0 = distr(gen);
            int i1 = i0 + 5;

            nq = i1 - i0;
            queries.resize(nq * d);
            for (int i = i0; i < i1; i++) {
                for (int j = 0; j < d; j++) {
                    queries[(i - i0) * d + j] = database[i * d + j];
                }
            }
        }

        { // searching the database
            int k = 5;
            printf("[%.3f s] Searching the %d nearest neighbors "
                "of %ld vectors in the index\n",
                elapsed() - t0,
                k,
                nq);

            std::vector<faiss::Index::idx_t> nns(k * nq);
            std::vector<float> dis(k * nq);

            printf("HI: %ld\n", nns.at(4));

            index.search(nq, queries.data(), k, dis.data(), nns.data());

            index.hnsw.print_neighbor_stats(1);
            
            printf("[%.3f s] Query results (vector ids, then distances):\n",
                elapsed() - t0);

            faiss::HNSWStats2 stat = index.hnsw.get_stats();

            //std::tuple<int,int> a = stat.get_max("ndis"); 

            for (int i = 0; i < nq; i++) {
                printf("query %2d: ", i);
                for (int j = 0; j < k; j++) {
                    printf("%7ld ", nns[j + i * k]);
                }
                printf("\n     dis: ");
                for (int j = 0; j < k; j++) {
                    printf("%7g ", dis[j + i * k]);
                }
                printf("\n");
            }

            printf("note that the nearest neighbor is not at "
                "distance 0 due to quantization errors\n");

            std::ofstream f;
            f.open("test-16.tsv", std::ios_base::app);

            printf("outputting to file\n");

            std::vector<int> levels = index.hnsw.levels; 

            for (int x = 0; x < stat.upper_level_stats.size(); x++){
                    // faiss::SearchStat temp = stat.stat_list.at(x);
                    // f << temp.search_stat["level"] << "\t";
                    // f << temp.search_stat["nstep"] << "\t";
                    // f << temp.search_stat["ndis"] << "\t";
                    // f << temp.search_stat["nchanged"] << "\t";
                    // f << temp.search_stat["nvisited"] << "\t";
                    // f << temp.search_stat["nignored"] << "\n";
                    faiss::UpperLevelSearchStat temp = stat.upper_level_stats.at(x);

                    //temp.query = nns[0 + x * k];

                    // if(temp.edge_hops_away == 1){
                    //     greedyError(temp, temp.query, d, database, index);
                    // }

                    //f << temp.query << "\t";
                    f << temp.global_minima << "\t";
                    f << temp.found_minima << "\t";
                    f << temp.dist_global_minima << "\t";
                    f << temp.dist_found_minima << "\t";
                    f << temp.edge_hops_away << "\t";
                    f << temp.level << "\n";
            }

            f.close();

            // for (int i = 0; i < 2000; i++){
            //     if (index.hnsw.levels[i] == 2 && index.hnsw.entry_point != i){
            //         printf("node value is %i\n", i);
            //         size_t begin, end;
            //         index.hnsw.neighbor_range(index.hnsw.entry_point, 2, &begin, &end);
            //         for (int i = begin; i < end; i++){
            //             printf("%i\n",index.hnsw.neighbors[i]);
            //         }
            //         printf("\n\n");
            //         index.hnsw.neighbor_range(index.hnsw.entry_point, 1, &begin, &end);
            //         for (int i = begin; i < end; i++){
            //             printf("%i\n",index.hnsw.neighbors[i]);
            //         }
            //         break;
            //     }
            // }
        }
        }
        catch(...){
            continue;
        }
    }

    return 0;
}
