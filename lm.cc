#include "scrf/lm.h"
#include "ebt/ebt.h"
#include <fstream>

namespace lm {

    std::vector<int> fst::vertices() const
    {
        std::vector<int> result;

        for (int i = 0; i < data->vertices; ++i) {
            result.push_back(i);
        }

        return result;
    }

    std::vector<int> fst::edges() const
    {
        std::vector<int> result;

        for (int i = 0; i < data->edges.size(); ++i) {
            result.push_back(i);
        }

        return result;
    }

    double fst::weight(int e) const
    {
        return data->edges.at(e).weight;
    }

    std::vector<int> const& fst::in_edges(int v) const
    {
        return data->in_edges.at(v);
    }

    std::unordered_map<std::string, std::vector<int>> const& fst::in_edges_map(int v) const
    {
        return data->in_edges_map.at(v);
    }

    std::vector<int> const& fst::out_edges(int v) const
    {
        return data->out_edges.at(v);
    }

    std::unordered_map<std::string, std::vector<int>> const& fst::out_edges_map(int v) const
    {
        return data->out_edges_map.at(v);
    }

    int fst::tail(int e) const
    {
        return data->edges.at(e).tail;
    }

    int fst::head(int e) const
    {
        return data->edges.at(e).head;
    }

    int fst::initial() const
    {
        return data->initial;
    }

    int fst::final() const
    {
        return data->final;
    }

    std::string const& fst::input(int e) const
    {
        return data->edges.at(e).label;
    }

    std::string const& fst::output(int e) const
    {
        return data->edges.at(e).label;
    }

    fst load_arpa_lm(std::istream& is)
    {
        std::string line;

        int n = 0;
        int max_n = 0;

        std::unordered_map<std::vector<std::string>, int> hist;

        fst_data result;
        result.vertices = 0;

        result.initial = 0;
        result.vertices += 1;
        hist[std::vector<std::string>{}] = 0;

        while (std::getline(is, line)) {
            if (line.size() == 0) {
                continue;
            } else if (ebt::startswith(line, "\\")
                    && ebt::endswith(line, "-grams:")) {
                auto parts = ebt::split(line.substr(1), "-");
                n = std::stoi(parts[0]);
            } else if (ebt::startswith(line, "ngram")) {
                max_n = std::max(max_n,
                    std::stoi(ebt::split(ebt::split(line, "=")[0])[1]));
            } else if (line == "\\data\\" || line == "\\end\\") {
                continue;
            } else {
                auto parts = ebt::split(line);

                double boff = 0;
                if (parts.size() > n + 1) {
                    boff = std::stod(parts.back());
                    parts.pop_back();
                }

                auto ngram = std::vector<std::string> { parts.begin() + 1, parts.end() };
                std::vector<std::string> tail_h { ngram.begin(), ngram.end() - 1 };

                if (!ebt::in(tail_h, hist)) {
                    hist[tail_h] = result.vertices;
                    result.vertices += 1;
                }

                std::vector<std::string> head_h;

                if (n == max_n) {
                    head_h = std::vector<std::string> { ngram.begin() + 1, ngram.end() };
                } else {
                    head_h = ngram;
                }

                if (!ebt::in(head_h, hist)) {
                    hist[head_h] = result.vertices;
                    result.vertices += 1;
                }

                edge_data e_data = { hist.at(tail_h), hist.at(head_h),
                    std::stod(parts.front()), ngram.back() };

                int e = result.edges.size();
                result.edges.push_back(e_data);

                if (int(hist.at(head_h)) > int(result.in_edges.size()) - 1) {
                    result.in_edges.resize(hist.at(head_h) + 1);
                    result.in_edges_map.resize(hist.at(head_h) + 1);
                }

                result.in_edges.at(hist.at(head_h)).push_back(e);
                result.in_edges_map.at(hist.at(head_h))[ngram.back()].push_back(e);

                if (hist.at(tail_h) > int(result.out_edges.size()) - 1) {
                    result.out_edges.resize(hist.at(tail_h) + 1);
                    result.out_edges_map.resize(hist.at(tail_h) + 1);
                }

                result.out_edges.at(hist.at(tail_h)).push_back(e);
                result.out_edges_map.at(hist.at(tail_h))[ngram.back()].push_back(e);

                if (boff != 0) {
                    std::vector<std::string> tail_h = head_h;
                    std::vector<std::string> head_h { tail_h.begin() + 1, tail_h.end() };

                    if (!ebt::in(head_h, hist)) {
                        hist[head_h] = result.vertices;
                        result.vertices += 1;
                    }

                    edge_data e_data = { hist.at(tail_h), hist.at(head_h),
                        boff, "<eps>" };

                    int e = result.edges.size();
                    result.edges.push_back(e_data);

                    if (int(hist.at(head_h)) > int(result.in_edges.size()) - 1) {
                        result.in_edges.resize(hist.at(head_h) + 1);
                        result.in_edges_map.resize(hist.at(head_h) + 1);
                    }

                    result.in_edges.at(hist.at(head_h)).push_back(e);
                    result.in_edges_map.at(hist.at(head_h))[std::string("<eps>")].push_back(e);

                    if (hist.at(tail_h) > int(result.out_edges.size()) - 1) {
                        result.out_edges.resize(hist.at(tail_h) + 1);
                        result.out_edges_map.resize(hist.at(tail_h) + 1);
                    }

                    result.out_edges.at(hist.at(tail_h)).push_back(e);
                    result.out_edges_map.at(hist.at(tail_h))[std::string("<eps>")].push_back(e);
                }
            }
        }

        result.initial = hist.at(std::vector<std::string>{});

        // TODO: dangerous
        result.final = hist.at({"</s>"});

        fst f;
        f.data = std::make_shared<fst_data>(std::move(result));

        return f;
    }

    fst load_arpa_lm(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_arpa_lm(ifs);
    }

}
