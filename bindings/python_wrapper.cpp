//
// Created by mingqi on 25-7-5.
//

// diskann_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <future>
#include <atomic>

#include "v2/index_merger.h"
#include "v2/merge_insert.h"
#include "aux_utils.h"
#include "utils.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "index.h"

namespace py = pybind11;

template<typename T, typename TagT = uint32_t>
class DiskANNIndex {
private:
    std::unique_ptr<diskann::MergeInsert<T, TagT>> merge_insert_;
    std::unique_ptr<diskann::Distance<T>> distance_;

    uint32_t max_pts_;
    uint32_t ndim_;
    uint32_t R_;
    uint32_t L_;
    uint32_t num_threads_;

    bool is_built_;
    bool save_index_as_one_file_;
    std::string working_folder_;
    std::string mem_prefix_;
    std::string base_prefix_;
    std::string merge_prefix_;

    std::mutex index_mutex_;
    std::atomic<bool> insertions_done_{true};
    std::atomic<bool> deletions_done_{true};

    // 临时文件管理
    std::string temp_data_file_;
    std::string temp_tags_file_;

    void create_temp_files() {
        temp_data_file_ = working_folder_ + "/temp_data.bin";
        temp_tags_file_ = working_folder_ + "/temp_tags.bin";
    }

    void save_data_to_file(const py::array_t<T>& data, const std::vector<TagT>& tags) {
        auto buf = data.request();
        uint32_t npts = buf.shape[0];
        uint32_t ndim = buf.shape[1];

        // 保存数据点
        std::ofstream data_writer(temp_data_file_, std::ios::binary);
        data_writer.write(reinterpret_cast<const char*>(&npts), sizeof(uint32_t));
        data_writer.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
        data_writer.write(reinterpret_cast<const char*>(buf.ptr),
                         npts * ndim * sizeof(T));
        data_writer.close();

        // 保存标签
        std::ofstream tags_writer(temp_tags_file_, std::ios::binary);
        uint32_t tag_count = tags.size();
        uint32_t tag_dim = 1;
        tags_writer.write(reinterpret_cast<const char*>(&tag_count), sizeof(uint32_t));
        tags_writer.write(reinterpret_cast<const char*>(&tag_dim), sizeof(uint32_t));
        tags_writer.write(reinterpret_cast<const char*>(tags.data()),
                         tag_count * sizeof(TagT));
        tags_writer.close();
    }

public:
    DiskANNIndex() : is_built_(false), save_index_as_one_file_(false) {
        working_folder_ = "./data/freshdiskann_working";
        mem_prefix_ = working_folder_ + "/mem_index";
        base_prefix_ = working_folder_ + "/base_index";
        merge_prefix_ = working_folder_ + "/merge_index";

        // 创建工作目录
        system(("mkdir -p " + working_folder_).c_str());
        create_temp_files();
    }

    ~DiskANNIndex() {
        // 清理临时文件
        // system(("rm -rf " + working_folder_).c_str());
    }

    void setup(uint32_t max_pts, uint32_t ndim, uint32_t R = 64,
               uint32_t L = 100, uint32_t num_threads = 1) {
        max_pts_ = max_pts;
        ndim_ = ndim;
        R_ = R;
        L_ = L;
        num_threads_ = num_threads;

        // 初始化距离函数
        if constexpr (std::is_same<T, float>::value) {
            distance_ = std::make_unique<diskann::DistanceL2>();
        } else if constexpr (std::is_same<T, uint8_t>::value) {
            distance_ = std::make_unique<diskann::DistanceL2UInt8>();
        } else if constexpr (std::is_same<T, int8_t>::value) {
            distance_ = std::make_unique<diskann::DistanceL2Int8>();
        }

        std::cout << "DiskANN setup completed: max_pts=" << max_pts_
                  << ", ndim=" << ndim_ << ", R=" << R_ << ", L=" << L_ << std::endl;
    }

    void build(const py::array_t<T>& data, uint32_t npts, const std::vector<TagT>& tags) {
        if (is_built_) {
            std::cout << "Index already built, skipping build phase" << std::endl;
            return;
        }

        std::lock_guard<std::mutex> lock(index_mutex_);

        auto buf = data.request();
        if (buf.ndim != 2 || buf.shape[1] != ndim_) {
            throw std::runtime_error("Data shape mismatch");
        }

        // 保存数据到临时文件
        save_data_to_file(data, tags);

        // 构建基础索引
        std::cout << "Building base disk index..." << std::endl;
        std::string build_params = std::to_string(R_) + " " + std::to_string(L_) +
                                  " " + std::to_string(3) + " " + std::to_string(3) +
                                  " " + std::to_string(num_threads_);

        bool success = diskann::build_disk_index<T>(
            temp_data_file_.c_str(),
            base_prefix_.c_str(),
            build_params.c_str(),
            diskann::Metric::L2,
            save_index_as_one_file_,
            temp_tags_file_.c_str()
        );

        if (!success) {
            throw std::runtime_error("Failed to build base disk index");
        }

        // 初始化MergeInsert
        diskann::Parameters params;
        params.Set<unsigned>("L_mem", L_);
        params.Set<unsigned>("R_mem", R_);
        params.Set<float>("alpha_mem", 1.2f);
        params.Set<unsigned>("L_disk", L_);
        params.Set<unsigned>("R_disk", R_);
        params.Set<float>("alpha_disk", 1.2f);
        params.Set<unsigned>("C", R_ * 2);
        params.Set<unsigned>("beamwidth", 2);
        params.Set<unsigned>("nodes_to_cache", 100);
        params.Set<unsigned>("num_search_threads", num_threads_);

        merge_insert_ = std::make_unique<diskann::MergeInsert<T, TagT>>(
            params, ndim_, mem_prefix_, base_prefix_, merge_prefix_,
            distance_.get(), diskann::Metric::L2, save_index_as_one_file_, working_folder_
        );

        is_built_ = true;
        std::cout << "Index built successfully" << std::endl;
    }

    void insert_concurrent(const py::array_t<T>& data, const py::array_t<TagT>& tags,
                          int insert_thread_count = 1) {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        auto data_buf = data.request();
        auto tags_buf = tags.request();

        if (data_buf.shape[0] != tags_buf.shape[0]) {
            throw std::runtime_error("Data and tags count mismatch");
        }

        uint32_t npts = data_buf.shape[0];
        uint32_t ndim = data_buf.shape[1];

        if (ndim != ndim_) {
            throw std::runtime_error("Data dimension mismatch");
        }

        T* data_ptr = static_cast<T*>(data_buf.ptr);
        TagT* tags_ptr = static_cast<TagT*>(tags_buf.ptr);

        std::cout << "Inserting " << npts << " points with " << insert_thread_count << " threads" << std::endl;

        // 并发插入
        std::vector<std::future<void>> futures;
        uint32_t points_per_thread = npts / insert_thread_count;

        for (int t = 0; t < insert_thread_count; ++t) {
            uint32_t start_idx = t * points_per_thread;
            uint32_t end_idx = (t == insert_thread_count - 1) ? npts : (t + 1) * points_per_thread;

            futures.push_back(std::async(std::launch::async, [this, data_ptr, tags_ptr, start_idx, end_idx, ndim]() {
                for (uint32_t i = start_idx; i < end_idx; ++i) {
                    T* point = data_ptr + i * ndim;
                    TagT tag = tags_ptr[i];

                    int result = merge_insert_->insert(point, tag);
                    if (result != 0) {
                        std::cout << "Failed to insert point " << i << " with tag " << tag << std::endl;
                    }
                }
            }));
        }

        // 等待所有线程完成
        for (auto& future : futures) {
            future.wait();
        }

        std::cout << "Insertion completed" << std::endl;
    }

    bool insert_single(const py::array_t<T>& point, TagT tag) {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        auto buf = point.request();
        if (buf.ndim != 1 || buf.shape[0] != ndim_) {
            throw std::runtime_error("Point dimension mismatch");
        }

        T* point_ptr = static_cast<T*>(buf.ptr);
        return merge_insert_->insert(point_ptr, tag) == 0;
    }

    void remove(const std::vector<TagT>& tags) {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        std::cout << "Removing " << tags.size() << " points" << std::endl;

        for (TagT tag : tags) {
//            TODO: 使用inplace_delete而不是lazy_delete以获得更好的性能;IP-dsiaknn与freshdiskann的主要区别就在
            merge_insert_->lazy_delete(tag);
        }

        std::cout << "Removal completed" << std::endl;
    }

    std::pair<std::vector<TagT>, std::vector<float>> query(const py::array_t<T>& query_point,
                                                           uint32_t k, uint32_t search_L = 0) {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        if (search_L == 0) {
            search_L = L_;
        }

        auto buf = query_point.request();
        if (buf.ndim != 1 || buf.shape[0] != ndim_) {
            throw std::runtime_error("Query point dimension mismatch");
        }

        T* query_ptr = static_cast<T*>(buf.ptr);

        std::vector<TagT> result_tags(k);
        std::vector<float> result_dists(k);
        diskann::QueryStats stats;

        merge_insert_->search_sync(query_ptr, k, search_L,
                                  result_tags.data(), result_dists.data(), &stats);

        return std::make_pair(result_tags, result_dists);
    }

    std::pair<py::array_t<TagT>, py::array_t<float>> batch_query(const py::array_t<T>& queries,
                                                                uint32_t k, int search_thread_count = 1) {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        auto buf = queries.request();
        if (buf.ndim != 2 || buf.shape[1] != ndim_) {
            throw std::runtime_error("Query data dimension mismatch");
        }

        uint32_t nqueries = buf.shape[0];
        T* queries_ptr = static_cast<T*>(buf.ptr);

        // 准备结果数组
        auto result_tags = py::array_t<TagT>(nqueries * k);
        auto result_dists = py::array_t<float>(nqueries * k);

        auto tags_buf = result_tags.request();
        auto dists_buf = result_dists.request();

        TagT* tags_ptr = static_cast<TagT*>(tags_buf.ptr);
        float* dists_ptr = static_cast<float*>(dists_buf.ptr);

        // 并发查询
        #pragma omp parallel for num_threads(search_thread_count)
        for (uint32_t i = 0; i < nqueries; ++i) {
            T* query_ptr = queries_ptr + i * ndim_;
            TagT* result_tags_ptr = tags_ptr + i * k;
            float* result_dists_ptr = dists_ptr + i * k;

            diskann::QueryStats stats;
            merge_insert_->search_sync(query_ptr, k, L_,
                                      result_tags_ptr, result_dists_ptr, &stats);
        }

        // 重新调整数组形状
        result_tags.resize({nqueries, k});
        result_dists.resize({nqueries, k});

        return std::make_pair(result_tags, result_dists);
    }

    void final_merge() {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        std::cout << "Performing final merge..." << std::endl;
        merge_insert_->final_merge();
        std::cout << "Final merge completed" << std::endl;
    }

    void save_index(const std::string& filepath) {
        if (!is_built_) {
            throw std::runtime_error("Index not built yet");
        }

        // 实现索引保存逻辑
        std::cout << "Saving index to " << filepath << std::endl;
        // 这里需要根据实际的DiskANN API来实现
    }

    void load_index(const std::string& filepath) {
        // 实现索引加载逻辑
        std::cout << "Loading index from " << filepath << std::endl;
        // 这里需要根据实际的DiskANN API来实现
        is_built_ = true;
    }

    // 获取索引统计信息
    std::map<std::string, uint32_t> get_stats() const {
        std::map<std::string, uint32_t> stats;
        stats["max_pts"] = max_pts_;
        stats["ndim"] = ndim_;
        stats["R"] = R_;
        stats["L"] = L_;
        stats["num_threads"] = num_threads_;
        stats["is_built"] = is_built_ ? 1 : 0;
        return stats;
    }
};

PYBIND11_MODULE(freshdiskann, m) {
    m.doc() = "FreshDiskANN Python bindings";

    // 导出float版本的索引
    py::class_<DiskANNIndex<float>>(m, "Index")
        .def(py::init<>())
        .def("setup", &DiskANNIndex<float>::setup,
             "Setup index parameters",
             py::arg("max_pts"), py::arg("ndim"), py::arg("R") = 64,
             py::arg("L") = 100, py::arg("num_threads") = 1)
        .def("build", &DiskANNIndex<float>::build,
             "Build index from data",
             py::arg("data"), py::arg("npts"), py::arg("tags"))
        .def("insert_concurrent", &DiskANNIndex<float>::insert_concurrent,
             "Insert points concurrently",
             py::arg("data"), py::arg("tags"), py::arg("insert_thread_count") = 1)
        .def("insert", &DiskANNIndex<float>::insert_single,
             "Insert single point",
             py::arg("point"), py::arg("tag"))
        .def("remove", &DiskANNIndex<float>::remove,
             "Remove points by tags",
             py::arg("tags"))
        .def("query", &DiskANNIndex<float>::query,
             "Query k nearest neighbors",
             py::arg("query_point"), py::arg("k"), py::arg("search_L") = 0)
        .def("batch_query", &DiskANNIndex<float>::batch_query,
             "Batch query k nearest neighbors",
             py::arg("queries"), py::arg("k"), py::arg("search_thread_count") = 1)
        .def("final_merge", &DiskANNIndex<float>::final_merge,
             "Perform final merge of pending operations")
        .def("save_index", &DiskANNIndex<float>::save_index,
             "Save index to file",
             py::arg("filepath"))
        .def("load_index", &DiskANNIndex<float>::load_index,
             "Load index from file",
             py::arg("filepath"))
        .def("get_stats", &DiskANNIndex<float>::get_stats,
             "Get index statistics");

}