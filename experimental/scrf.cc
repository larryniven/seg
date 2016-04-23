#include "scrf/experimental/scrf.h"
#include <istream>
#include <fstream>
#include <cassert>
#include "ebt/ebt.h"
#include "opt/opt.h"
#include "la/la.h"

namespace scrf {

    dense_vec load_dense_vec(std::istream& is)
    {
        dense_vec result;
        std::string line;

        result = dense_vec { ebt::json::json_parser<
            std::vector<la::vector<double>>>().parse(is) };
        std::getline(is, line);

        return result;
    }

    dense_vec load_dense_vec(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_dense_vec(ifs);
    }

    void save_vec(dense_vec const& v, std::ostream& os)
    {
        os << v.class_vec << std::endl;
    }

    void save_vec(dense_vec const& v, std::string filename)
    {
        std::ofstream ofs { filename };
        save_vec(v, ofs);
    }

    double dot(dense_vec const& p1, dense_vec const& p2)
    {
        if (p1.class_vec.size() == 0 || p2.class_vec.size() == 0) {
            return 0;
        }

        double sum = 0;

        for (int i = 0; i < p2.class_vec.size(); ++i) {
            if (p1.class_vec[i].size() == 0 || p2.class_vec[i].size() == 0) {
                continue;
            }

            sum += la::dot(p1.class_vec[i], p2.class_vec[i]);
        }

        return sum;
    }

    void iadd(dense_vec& p1, dense_vec const& p2)
    {
        if (p1.class_vec.size() == 0) {
            p1.class_vec.resize(p2.class_vec.size());
        }

        for (int i = 0; i < p2.class_vec.size(); ++i) {
            auto& v = p1.class_vec[i];
            auto& u = p2.class_vec[i];

            if (u.size() == 0) {
                continue;
            }

            if (v.size() == 0) { 
                v.resize(u.size());
            } else {
                assert(v.size() == u.size());
            }

            la::iadd(v, u);
        }
    }

    void isub(dense_vec& p1, dense_vec const& p2)
    {
        if (p1.class_vec.size() == 0) {
            p1.class_vec.resize(p2.class_vec.size());
        }

        for (int i = 0; i < p2.class_vec.size(); ++i) {
            auto& v = p1.class_vec[i];
            auto& u = p2.class_vec[i];

            if (u.size() == 0) {
                continue;
            }

            if (v.size() == 0) { 
                v.resize(u.size());
            } else {
                assert(v.size() == u.size());
            }

            la::isub(v, u);
        }
    }

    void imul(dense_vec& p, double c)
    {
        if (c == 0) {
            p.class_vec.clear();
        }

        for (int i = 0; i < p.class_vec.size(); ++i) {
            if (p.class_vec[i].size() == 0) {
                continue;
            }

            la::imul(p.class_vec[i], c);
        }
    }

    void adagrad_update(dense_vec& param, dense_vec const& grad,
        dense_vec& accu_grad_sq, double step_size)
    {
        if (accu_grad_sq.class_vec.size() == 0) {
            accu_grad_sq.class_vec.resize(grad.class_vec.size());
        }

        if (param.class_vec.size() == 0) {
            param.class_vec.resize(grad.class_vec.size());
        }

        for (int i = 0; i < grad.class_vec.size(); ++i) {
            if (grad.class_vec[i].size() == 0) {
                continue;
            }

            param.class_vec[i].resize(grad.class_vec[i].size());
            accu_grad_sq.class_vec[i].resize(grad.class_vec[i].size());

            opt::adagrad_update(param.class_vec[i], grad.class_vec[i],
                accu_grad_sq.class_vec[i], step_size);
        }
    }

    void rmsprop_update(dense_vec& param, dense_vec const& grad,
        dense_vec& accu_grad_sq, double decay, double step_size)
    {
        if (accu_grad_sq.class_vec.size() == 0) {
            accu_grad_sq.class_vec.resize(grad.class_vec.size());
        }

        if (param.class_vec.size() == 0) {
            param.class_vec.resize(grad.class_vec.size());
        }

        for (int i = 0; i < grad.class_vec.size(); ++i) {
            if (grad.class_vec[i].size() == 0) {
                continue;
            }

            param.class_vec[i].resize(grad.class_vec[i].size());
            accu_grad_sq.class_vec[i].resize(grad.class_vec[i].size());

            opt::rmsprop_update(param.class_vec[i], grad.class_vec[i],
                accu_grad_sq.class_vec[i], decay, step_size);
        }
    }

    double dot(sparse_vec const& u, sparse_vec const& v)
    {
         double sum = 0;
         for (auto& p: u.class_vec) {
             if (ebt::in(p.first, v.class_vec)) {
                 sum += la::dot(p.second, v.class_vec.at(p.first));
             }
         }
         return sum;
    }

    sparse_vec load_sparse_vec(std::istream& is)
    {
        ebt::json::json_parser<std::unordered_map<std::string, la::vector<double>>> parser;

        sparse_vec result { parser.parse(is) };

        return result;
    }

    sparse_vec load_sparse_vec(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_sparse_vec(ifs);
    }

    void save_vec(sparse_vec const& v, std::ostream& os)
    {
        os << v.class_vec << std::endl;
    }

    void save_vec(sparse_vec const& v, std::string filename)
    {
        std::ofstream ofs { filename };

        save_vec(v, ofs);
    }

    void iadd(sparse_vec& u, sparse_vec const& v)
    {
        for (auto& p: v.class_vec) {
            auto& k = u.class_vec[p.first];
            if (k.size() == 0) {
                k.resize(p.second.size());
            }
            la::iadd(k, p.second);
        }
    }

    void isub(sparse_vec& u, sparse_vec const& v)
    {
        for (auto& p: v.class_vec) {
            auto& k = u.class_vec[p.first];
            if (k.size() == 0) {
                k.resize(p.second.size());
            }
            la::isub(k, p.second);
        }
    }

    void imul(sparse_vec& u, double c)
    {
        for (auto& p: u.class_vec) {
            la::imul(p.second, c);
        }
    }

    void adagrad_update(sparse_vec& theta, sparse_vec const& grad,
        sparse_vec& accu_grad_sq, double step_size)
    {
        for (auto& p: grad.class_vec) {
            if (theta.class_vec[p.first].size() == 0) {
                theta.class_vec[p.first].resize(p.second.size());
            }

            if (accu_grad_sq.class_vec[p.first].size() == 0) {
                accu_grad_sq.class_vec[p.first].resize(p.second.size());
            }

            opt::adagrad_update(theta.class_vec.at(p.first), p.second,
                accu_grad_sq.class_vec.at(p.first), step_size);
        }
    }

    void rmsprop_update(sparse_vec& theta, sparse_vec const& grad,
        sparse_vec& accu_grad_sq, double decay, double step_size)
    {
        for (auto& p: grad.class_vec) {
            if (theta.class_vec[p.first].size() == 0) {
                theta.class_vec[p.first].resize(p.second.size());
            }

            if (accu_grad_sq.class_vec[p.first].size() == 0) {
                accu_grad_sq.class_vec[p.first].resize(p.second.size());
            }

            opt::rmsprop_update(theta.class_vec.at(p.first), p.second,
                accu_grad_sq.class_vec.at(p.first), decay, step_size);
        }
    }

    std::unordered_map<std::string, int> load_label_id(std::string filename)
    {
        std::unordered_map<std::string, int> result;
        std::string line;
        std::ifstream ifs { filename };

        result["<eps>"] = 0;
    
        int i = 1;
        while (std::getline(ifs, line)) {
            result[line] = i;
            ++i;
        }
    
        return result;
    }

    std::pair<int, int> get_dim(std::string feat)
    {
        std::vector<std::string> parts = ebt::split(feat, ":");
        int start_dim = -1;
        int end_dim = -1;
        if (parts.size() == 2) {
            std::vector<std::string> indices = ebt::split(parts.back(), "-");
            start_dim = std::stoi(indices.at(0));
            end_dim = std::stoi(indices.at(1));
        }

        return std::make_pair(start_dim, end_dim);
    }

}
