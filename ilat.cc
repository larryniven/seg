#include "scrf/experimental/ilat.h"
#include "ebt/ebt.h"
#include <algorithm>
#include <cassert>
#include <fstream>

namespace ilat {

    bool operator==(vertex_data const& v1, vertex_data const& v2)
    {
        return v1.time == v2.time;
    }

    bool operator==(edge_data const& e1, edge_data const& e2)
    {
        return e1.input == e2.input && e1.output == e2.output
            && e1.tail == e2.tail && e1.head == e2.head
            && e1.weight == e2.weight;
    }

    void add_vertex(fst_data& data, int v, vertex_data v_data)
    {
        if (!ebt::in(v, data.vertex_set)) {
            data.vertex_set.insert(v);
            data.vertex_indices.push_back(v);

            int size = std::max<int>(v + 1, data.vertices.size());

            data.vertices.resize(size);
            data.vertices[v] = v_data;
            data.in_edges.resize(size);
            data.out_edges.resize(size);
            data.in_edges_map.resize(size);
            data.out_edges_map.resize(size);
            data.vertex_attrs.resize(size);
        } else {
            assert(data.vertices[v] == v_data);
        }
    }

    void add_edge(fst_data& data, int e, edge_data e_data)
    {
        assert(ebt::in(e_data.head, data.vertex_set));
        assert(ebt::in(e_data.tail, data.vertex_set));

        if (!ebt::in(e, data.edge_set)) {
            data.edge_set.insert(e);
            data.edge_indices.push_back(e);

            int size = std::max<int>(e + 1, data.edges.size());

            data.edges.resize(size);
            data.edges[e] = e_data;
            data.in_edges[e_data.head].push_back(e);
            data.out_edges[e_data.tail].push_back(e);
            data.in_edges_map[e_data.head].resize(data.symbol_id->size());
            data.in_edges_map[e_data.head][e_data.input].push_back(e);
            data.out_edges_map[e_data.tail].resize(data.symbol_id->size());
            data.out_edges_map[e_data.tail][e_data.input].push_back(e);
            data.edge_attrs.resize(size);
            data.feats.resize(size);
        } else {
            assert(data.edges[e] == e_data);
        }
    }

    std::vector<int> const& fst::vertices() const
    {
        return data->vertex_indices;
    }

    std::vector<int> const& fst::edges() const
    {
        return data->edge_indices;
    }

    double fst::weight(int e) const
    {
        return data->edges.at(e).weight;
    }

    std::vector<int> const& fst::in_edges(int v) const
    {
        return data->in_edges.at(v);
    }

    std::vector<int> const& fst::out_edges(int v) const
    {
        return data->out_edges.at(v);
    }

    std::vector<std::vector<int>> const& fst::in_edges_map(int v) const
    {
        return data->in_edges_map.at(v);
    }

    std::vector<std::vector<int>> const& fst::out_edges_map(int v) const
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

    std::vector<int> const& fst::initials() const
    {
        return data->initials;
    }

    std::vector<int> const& fst::finals() const
    {
        return data->finals;
    }

    int const& fst::input(int e) const
    {
        return data->edges.at(e).input;
    }

    int const& fst::output(int e) const
    {
        return data->edges.at(e).output;
    }

    long fst::time(int v) const
    {
        return data->vertices.at(v).time;
    }

    fst load_lattice(std::istream& is, std::unordered_map<std::string, int> const& symbol_id)
    {
        fst_data result;

        result.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(symbol_id);
        
        std::vector<std::string> id_symbol;
        id_symbol.resize(symbol_id.size());
        for (auto& p: symbol_id) {
            id_symbol[p.second] = p.first;
        }
        result.id_symbol = std::make_shared<std::vector<std::string>>(id_symbol);

        std::string line;

        // read filename
        if (!std::getline(is, line)) {
            fst f;
            f.data = std::make_shared<fst_data>(result);
            return f;
        }

        result.name = line;

        int min_time = std::numeric_limits<int>::max();
        std::unordered_set<int> initials;

        int max_time = std::numeric_limits<int>::min();
        std::unordered_set<int> finals;

        while (std::getline(is, line) && line != "#") {
            if (line == ".") {
                std::cout << "\"#\" was not found between vertices and edges." << std::endl;
                exit(1);
            }

            auto parts = ebt::split(line);

            int v = std::stoi(parts[0]);

            std::unordered_map<std::string, std::string> attr_map;
            std::vector<std::pair<std::string, std::string>> attrs;
            auto pairs = ebt::split(parts[1], ";");
            for (auto& p: pairs) {
                auto pair = ebt::split(p, "=");
                attr_map[pair[0]] = pair[1];
                attrs.push_back(std::make_pair(pair[0], pair[1]));
            }

            ilat::add_vertex(result, v, ilat::vertex_data { std::stoi(attr_map.at("time")) });

            result.vertex_attrs[v] = attrs;
        }

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            int tail = std::stoi(parts[0]);
            int head = std::stoi(parts[1]);

            std::unordered_map<std::string, std::string> attr_map;
            std::vector<std::pair<std::string, std::string>> attr;
            std::vector<double> feats;

            auto pairs = ebt::split(parts[2], ";");
            for (auto& p: pairs) {
                auto pair = ebt::split(p, "=");

                if (pair[0] == "feat") {
                    auto parts = ebt::split(pair[1], ",");
                    feats.resize(parts.size());
                    std::transform(parts.begin(), parts.end(), feats.begin(),
                        [](std::string const& s) { return std::stod(s); });
                } else {
                    attr_map[pair[0]] = pair[1];
                    attr.push_back(std::make_pair(pair[0], pair[1]));
                }
            }

            double weight = 0;
            if (ebt::in(std::string("weight"), attr_map)) {
                weight = std::stod(attr_map.at("weight"));
            }

            std::string label = attr_map.at("label");

            int e = int(result.edges.size());
            add_edge(result, e, edge_data { tail, head, weight,
                symbol_id.at(label), symbol_id.at(label) });

            result.edge_attrs[e] = attr;
            result.feats[e] = feats;

            if (max_time < result.vertices.at(head).time) {
                max_time = result.vertices.at(head).time;
                finals.clear();
                finals.insert(head);
            } else if (max_time == result.vertices.at(head).time) {
                finals.insert(head);
            }

            if (result.vertices.at(tail).time < min_time) {
                min_time = result.vertices.at(tail).time;
                initials.clear();
                initials.insert(tail);
            } else if (min_time == result.vertices.at(tail).time) {
                initials.insert(tail);
            }
        }

        result.initials = std::vector<int> { initials.begin(), initials.end() };
        result.finals = std::vector<int> { finals.begin(), finals.end() };

        fst f;
        f.data = std::make_shared<fst_data>(std::move(result));

        return f;
    }

    fst add_eps_loops(fst f, int label)
    {
        fst_data& data = *(f.data);

        for (int i = 0; i < data.vertices.size(); ++i) {
            int e = int(data.edges.size());
            add_edge(data, e, edge_data {i, i, 0, label, label});
        }

        return f;
    }

    std::shared_ptr<fst> ilat_path_maker::operator()(std::vector<int> const& edges, fst const& f) const
    {
        fst_data data;

        data.symbol_id = f.data->symbol_id;
        data.id_symbol = f.data->id_symbol;

        for (auto& e: edges) {
            add_vertex(data, f.tail(e), vertex_data { f.time(f.tail(e)) });
            add_vertex(data, f.head(e), vertex_data { f.time(f.head(e)) });
            add_edge(data, e, edge_data { f.tail(e), f.head(e), f.weight(e), f.input(e), f.output(e) });
        }

        data.name = f.data->name;

        for (auto& v: f.initials()) {
            if (ebt::in(v, data.vertex_set)) {
                data.initials.push_back(v);
            }
        }

        for (auto& v: f.finals()) {
            if (ebt::in(v, data.vertex_set)) {
                data.finals.push_back(v);
            }
        }

        for (auto& v: data.vertex_indices) {
            data.vertex_attrs[v] = f.data->vertex_attrs[v];
        }

        for (auto& e: data.edge_indices) {
            data.edge_attrs[e] = f.data->edge_attrs[e];
            data.feats[e] = f.data->feats[e];
        }

        fst result;
        result.data = std::make_shared<fst_data>(data);

        return std::make_shared<fst>(result);
    }

    fst load_arpa_lm(std::istream& is,
        std::unordered_map<std::string, int> const& symbol_id)
    {
        std::string line;

        int n = 0;
        int max_n = 0;

        std::unordered_map<std::vector<std::string>, int> hist;

        fst_data result;

        result.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(symbol_id);

        std::vector<std::string> id_symbol;
        id_symbol.resize(symbol_id.size());
        for (auto& p: symbol_id) {
            id_symbol[p.second] = p.first;
        }
        result.id_symbol = std::make_shared<std::vector<std::string>>(id_symbol);

        result.initials.push_back(0);
        add_vertex(result, 0, vertex_data { 0 });

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

                double boff = 1;
                if (parts.size() > n + 1) {
                    boff = std::stod(parts.back());
                    parts.pop_back();
                }

                auto ngram = std::vector<std::string> { parts.begin() + 1, parts.end() };
                std::vector<std::string> tail_h { ngram.begin(), ngram.end() - 1 };

                if (!ebt::in(tail_h, hist)) {
                    int v = result.vertices.size();
                    hist[tail_h] = v;
                    add_vertex(result, v, vertex_data { 0 });
                }

                std::vector<std::string> head_h;

                if (n == max_n) {
                    head_h = std::vector<std::string> { ngram.begin() + 1, ngram.end() };
                } else {
                    head_h = ngram;
                }

                if (!ebt::in(head_h, hist)) {
                    int v = result.vertices.size();
                    hist[head_h] = v;
                    add_vertex(result, v, vertex_data { 0 });
                }

                edge_data e_data = { hist.at(tail_h), hist.at(head_h),
                    std::stod(parts.front()), symbol_id.at(ngram.back()), symbol_id.at(ngram.back()) };

                int e = result.edges.size();

                add_edge(result, e, e_data);

                if (boff != 1) {
                    std::vector<std::string> tail_h = head_h;
                    std::vector<std::string> head_h { tail_h.begin() + 1, tail_h.end() };

                    if (!ebt::in(head_h, hist)) {
                        int v = result.vertices.size();
                        hist[head_h] = v;
                        add_vertex(result, v, vertex_data { 0 });
                    }

                    edge_data e_data = { hist.at(tail_h), hist.at(head_h),
                        boff, symbol_id.at("<eps>"), symbol_id.at("<eps>") };

                    int e = result.edges.size();

                    add_edge(result, e, e_data);
                }
            }
        }

        for (auto& p: hist) {
            result.vertex_attrs[p.second].push_back(
                std::make_pair("history", ebt::join(p.first, "_")));
        }

        for (auto& v: result.vertex_indices) {
            result.finals.push_back(v);
        }

        fst f;
        f.data = std::make_shared<fst_data>(std::move(result));

        return f;
    }

    fst load_arpa_lm(std::string filename,
        std::unordered_map<std::string, int> const& symbol_id)
    {
        std::ifstream ifs { filename };
        return load_arpa_lm(ifs, symbol_id);
    }

    lazy_pair_mode1::lazy_pair_mode1(ilat::fst fst1, ilat::fst fst2)
        : fst1_(fst1), fst2_(fst2)
        , vertices_cache(nullptr), edges_cache(nullptr)
        , initials_cache(nullptr), finals_cache(nullptr)
        , in_edges_vertex(nullptr), in_edges_cache(nullptr)
        , out_edges_vertex(nullptr), out_edges_cache(nullptr)
    {
    }

    std::vector<lazy_pair_mode1::vertex> const& lazy_pair_mode1::vertices() const
    {
        if (vertices_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;
            for (int u: fst1_.vertices()) {
                for (int v: fst2_.vertices()) {
                    result.push_back(std::make_tuple(u, v));
                }
            }

            vertices_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *vertices_cache;
    }

    std::vector<lazy_pair_mode1::edge> const& lazy_pair_mode1::edges() const
    {
        if (edges_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;

            for (int e1: fst1_.edges()) {
                for (int e2: fst2_.edges()) {
                    if (fst1_.output(e1) == fst2_.input(e2)) {
                        result.push_back(std::make_tuple(e1, e2));
                    }
                }
            }

            edges_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *edges_cache;
    }

    double lazy_pair_mode1::weight(lazy_pair_mode1::edge e) const
    {
        return fst1_.weight(std::get<0>(e)) + fst2_.weight(std::get<1>(e));
    }

    std::vector<lazy_pair_mode1::edge> const& lazy_pair_mode1::in_edges(lazy_pair_mode1::vertex v) const
    {
        if (in_edges_vertex == nullptr || *in_edges_vertex != v) {
            in_edges_vertex = std::make_shared<std::tuple<int, int>>(v);

            std::vector<std::tuple<int, int>> result;

            for (int e1: fst1_.in_edges(std::get<0>(v))) {
                auto& edge_map = fst2_.in_edges_map(std::get<1>(v));

                if (edge_map.size() == 0) {
                    continue;
                }

                for (auto& e2: edge_map.at(fst1_.output(e1))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            in_edges_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *in_edges_cache;
    }

    std::vector<lazy_pair_mode1::edge> const& lazy_pair_mode1::out_edges(lazy_pair_mode1::vertex v) const
    {
        if (out_edges_vertex == nullptr || *out_edges_vertex != v) {
            out_edges_vertex = std::make_shared<std::tuple<int, int>>(v);

            std::vector<std::tuple<int, int>> result;

            for (int e1: fst1_.out_edges(std::get<0>(v))) {
                auto& edge_map = fst2_.out_edges_map(std::get<1>(v));

                if (edge_map.size() == 0) {
                    continue;
                }

                for (auto& e2: edge_map.at(fst1_.output(e1))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            out_edges_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *out_edges_cache;
    }

    lazy_pair_mode1::vertex lazy_pair_mode1::tail(lazy_pair_mode1::edge e) const
    {
        return std::make_tuple(fst1_.tail(std::get<0>(e)), fst2_.tail(std::get<1>(e)));
    }

    lazy_pair_mode1::vertex lazy_pair_mode1::head(lazy_pair_mode1::edge e) const
    {
        return std::make_tuple(fst1_.head(std::get<0>(e)), fst2_.head(std::get<1>(e)));
    }

    std::vector<lazy_pair_mode1::vertex> const& lazy_pair_mode1::initials() const
    {
        if (initials_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;

            for (int i: fst1_.initials()) {
                for (int j: fst2_.initials()) {
                    result.push_back(std::make_tuple(i, j));
                }
            }

            initials_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *initials_cache;
    }

    std::vector<lazy_pair_mode1::vertex> const& lazy_pair_mode1::finals() const
    {
        if (finals_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;

            for (int i: fst1_.finals()) {
                for (int j: fst2_.finals()) {
                    result.push_back(std::make_tuple(i, j));
                }
            }

            finals_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *finals_cache;
    }

    int const& lazy_pair_mode1::input(lazy_pair_mode1::edge e) const
    {
        return fst1_.input(std::get<0>(e));
    }

    int const& lazy_pair_mode1::output(lazy_pair_mode1::edge e) const
    {
        return fst2_.output(std::get<1>(e));
    }

    long lazy_pair_mode1::time(lazy_pair_mode1::vertex v) const
    {
        return fst1_.time(std::get<0>(v));
    }

    ilat::fst const& lazy_pair_mode1::fst1() const
    {
        return fst1_;
    }

    ilat::fst const& lazy_pair_mode1::fst2() const
    {
        return fst2_;
    }

    lazy_pair_mode2::lazy_pair_mode2(ilat::fst fst1, ilat::fst fst2)
        : fst1_(fst1), fst2_(fst2)
        , vertices_cache(nullptr), edges_cache(nullptr)
        , initials_cache(nullptr), finals_cache(nullptr)
        , in_edges_vertex(nullptr), in_edges_cache(nullptr)
        , out_edges_vertex(nullptr), out_edges_cache(nullptr)
    {
    }

    std::vector<lazy_pair_mode2::vertex> const& lazy_pair_mode2::vertices() const
    {
        if (vertices_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;
            for (int u: fst1_.vertices()) {
                for (int v: fst2_.vertices()) {
                    result.push_back(std::make_tuple(u, v));
                }
            }

            vertices_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *vertices_cache;
    }

    std::vector<lazy_pair_mode2::edge> const& lazy_pair_mode2::edges() const
    {
        if (edges_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;

            for (int e1: fst1_.edges()) {
                for (int e2: fst2_.edges()) {
                    if (fst1_.output(e1) == fst2_.input(e2)) {
                        result.push_back(std::make_tuple(e1, e2));
                    }
                }
            }

            edges_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *edges_cache;
    }

    double lazy_pair_mode2::weight(lazy_pair_mode2::edge e) const
    {
        return fst1_.weight(std::get<0>(e)) + fst2_.weight(std::get<1>(e));
    }

    std::vector<lazy_pair_mode2::edge> const& lazy_pair_mode2::in_edges(lazy_pair_mode2::vertex v) const
    {
        if (in_edges_vertex == nullptr || *in_edges_vertex != v) {
            in_edges_vertex = std::make_shared<std::tuple<int, int>>(v);

            std::vector<std::tuple<int, int>> result;

            for (int e2: fst2_.in_edges(std::get<1>(v))) {
                auto& edge_map = fst1_.in_edges_map(std::get<0>(v));

                if (edge_map.size() == 0) {
                    continue;
                }

                for (auto& e1: edge_map.at(fst2_.output(e2))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            in_edges_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *in_edges_cache;
    }

    std::vector<lazy_pair_mode2::edge> const& lazy_pair_mode2::out_edges(lazy_pair_mode2::vertex v) const
    {
        if (out_edges_vertex == nullptr || *out_edges_vertex != v) {
            out_edges_vertex = std::make_shared<std::tuple<int, int>>(v);

            std::vector<std::tuple<int, int>> result;

            for (int e2: fst2_.out_edges(std::get<1>(v))) {
                auto& edge_map = fst1_.out_edges_map(std::get<0>(v));

                if (edge_map.size() == 0) {
                    continue;
                }

                for (auto& e1: edge_map.at(fst2_.output(e2))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            out_edges_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *out_edges_cache;
    }

    lazy_pair_mode2::vertex lazy_pair_mode2::tail(lazy_pair_mode2::edge e) const
    {
        return std::make_tuple(fst1_.tail(std::get<0>(e)), fst2_.tail(std::get<1>(e)));
    }

    lazy_pair_mode2::vertex lazy_pair_mode2::head(lazy_pair_mode2::edge e) const
    {
        return std::make_tuple(fst1_.head(std::get<0>(e)), fst2_.head(std::get<1>(e)));
    }

    std::vector<lazy_pair_mode2::vertex> const& lazy_pair_mode2::initials() const
    {
        if (initials_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;

            for (int i: fst1_.initials()) {
                for (int j: fst2_.initials()) {
                    result.push_back(std::make_tuple(i, j));
                }
            }

            initials_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *initials_cache;
    }

    std::vector<lazy_pair_mode2::vertex> const& lazy_pair_mode2::finals() const
    {
        if (finals_cache == nullptr) {
            std::vector<std::tuple<int, int>> result;

            for (int i: fst1_.finals()) {
                for (int j: fst2_.finals()) {
                    result.push_back(std::make_tuple(i, j));
                }
            }

            finals_cache = std::make_shared<std::vector<std::tuple<int, int>>>(
                std::move(result));
        }

        return *finals_cache;
    }

    int const& lazy_pair_mode2::input(lazy_pair_mode2::edge e) const
    {
        return fst1_.input(std::get<0>(e));
    }

    int const& lazy_pair_mode2::output(lazy_pair_mode2::edge e) const
    {
        return fst2_.output(std::get<1>(e));
    }

    long lazy_pair_mode2::time(lazy_pair_mode2::vertex v) const
    {
        return fst1_.time(std::get<0>(v));
    }

    ilat::fst const& lazy_pair_mode2::fst1() const
    {
        return fst1_;
    }

    ilat::fst const& lazy_pair_mode2::fst2() const
    {
        return fst2_;
    }

    std::shared_ptr<pair_fst> pair_fst_path_maker::operator()(
        std::vector<pair_fst::edge> const& edges, pair_fst const& f) const
    {
        pair_data data;

        std::unordered_set<composed_pair::vertex> vertex_set;

        for (auto& e: edges) {
            vertex_set.insert(f.tail(e));
            vertex_set.insert(f.head(e));
            data.edges.push_back(e);
            data.in_edges[f.head(e)].push_back(e);
            data.out_edges[f.tail(e)].push_back(e);
        }

        for (auto& v: f.initials()) {
            if (ebt::in(v, vertex_set)) {
                data.initials.push_back(v);
            }
        }

        for (auto& v: f.finals()) {
            if (ebt::in(v, vertex_set)) {
                data.initials.push_back(v);
            }
        }
        composed_pair path;

        path.fst1_ = f.fst1();
        path.fst2_ = f.fst2();
        path.data = std::make_shared<pair_data>(data);

        return std::make_shared<composed_pair>(path);
    }

    std::vector<composed_pair::vertex> const& composed_pair::vertices() const
    {
        return data->vertices;
    }

    std::vector<composed_pair::edge> const& composed_pair::edges() const
    {
        return data->edges;
    }

    double composed_pair::weight(composed_pair::edge e) const
    {
        return fst1_.weight(std::get<0>(e)) + fst2_.weight(std::get<1>(e));
    }

    std::vector<composed_pair::edge> const& composed_pair::in_edges(composed_pair::vertex v) const
    {
        if (ebt::in(v, data->in_edges)) {
            return data->in_edges.at(v);
        } else {
            return data->empty;
        }
    }

    std::vector<composed_pair::edge> const& composed_pair::out_edges(composed_pair::vertex v) const
    {
        if (ebt::in(v, data->out_edges)) {
            return data->out_edges.at(v);
        } else {
            return data->empty;
        }
    }

    composed_pair::vertex composed_pair::tail(composed_pair::edge e) const
    {
        return std::make_tuple(fst1_.tail(std::get<0>(e)), fst2_.tail(std::get<1>(e)));
    }

    composed_pair::vertex composed_pair::head(composed_pair::edge e) const
    {
        return std::make_tuple(fst1_.head(std::get<0>(e)), fst2_.head(std::get<1>(e)));
    }

    std::vector<composed_pair::vertex> const& composed_pair::initials() const
    {
        return data->initials;
    }

    std::vector<composed_pair::vertex> const& composed_pair::finals() const
    {
        return data->finals;
    }

    int const& composed_pair::input(composed_pair::edge e) const
    {
        return fst1_.input(std::get<0>(e));
    }

    int const& composed_pair::output(composed_pair::edge e) const
    {
        return fst2_.output(std::get<1>(e));
    }

    long composed_pair::time(composed_pair::vertex v) const
    {
        return fst1_.time(std::get<0>(v));
    }

    ilat::fst const& composed_pair::fst1() const
    {
        return fst1_;
    }

    ilat::fst const& composed_pair::fst2() const
    {
        return fst2_;
    }

}

namespace fst {

    int edge_trait<int>::null = -1;

    std::tuple<int, int> edge_trait<std::tuple<int, int>>::null = std::make_tuple(-1, -1);

    int symbol_trait<int>::eps = 0;

}

