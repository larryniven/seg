
namespace ctc {

    template <class fst_t>
    void beam_search<fst_t>::search(fst_t const& f, typename fst_t::output_symbol blk, int topk)
    {
        auto less = [&](extra_data const& d1, extra_data const& d2) {
            return path_score[std::make_pair(d1.ends_blank, d1.seq_id)]
                > path_score[std::make_pair(d2.ends_blank, d2.seq_id)];
        };

        std::unordered_set<typename fst_t::vertex> finals;
        for (auto& vf: f.finals()) {
            finals.insert(vf);
        }

        id_seq.push_back(std::vector<int>());
        seq_id[std::vector<int>()] = 0;
        path_score[std::make_pair(true, 0)] = 0;

        for (auto& i: f.initials()) {
            heap.push_back(extra_data {i, true, 0});
        }
        std::make_heap(heap.begin(), heap.end(), less);

        bool dirty = true;

        double inf = std::numeric_limits<double>::infinity();

        while (heap.size() > 0 && dirty) {

            dirty = false;

            std::vector<extra_data> old_heap;
            std::swap(old_heap, heap);

            std::unordered_set<std::pair<int, bool>> heap_set;

            std::vector<std::vector<int>> old_id_seq;
            std::swap(old_id_seq, id_seq);

            std::unordered_map<std::vector<int>, int> old_seq_id;
            std::swap(old_seq_id, seq_id);

            std::unordered_map<std::pair<bool, int>, double> old_path_score;
            std::swap(old_path_score, path_score);

            auto reg = [&](std::vector<int> const& seq) {
                int s;

                if (!ebt::in(seq, seq_id)) {
                    s = seq_id.size();
                    seq_id[seq] = s;
                    id_seq.push_back(seq);
                } else {
                    s = seq_id.at(seq);
                }

                return s;
            };

            auto get_score = [&](std::unordered_map<std::pair<bool, int>, double> const& score,
                    bool ends_blank, int seq_id) {
                std::pair<bool, int> k {ends_blank, seq_id};
                if (ebt::in(k, score)) {
                    return score.at(k);
                } else {
                    return -inf;
                }
            };

            for (auto& t: old_heap) {
                if (ebt::in(t.vertex, finals)) {
                    int s = reg(old_id_seq[t.seq_id]);
                    if (!ebt::in(std::make_pair(true, s), path_score)) {
                        heap.push_back(extra_data {t.vertex, t.ends_blank, s});
                        std::push_heap(heap.begin(), heap.end(), less);
                    }
                    path_score[std::make_pair(t.ends_blank, s)]
                        = ebt::log_add(get_score(path_score, t.ends_blank, s),
                            old_path_score.at(std::make_pair(t.ends_blank, t.seq_id)));
                    continue;
                }

                for (auto& e: f.out_edges(t.vertex)) {
                    if (t.ends_blank && f.output(e) == blk) {
                        int s = reg(old_id_seq[t.seq_id]);
                        if (!ebt::in(std::make_pair(true, s), path_score)) {
                            heap.push_back(extra_data {f.head(e), true, s});
                        }
                        path_score[std::make_pair(true, s)]
                            = ebt::log_add(get_score(path_score, true, s),
                                old_path_score.at(std::make_pair(t.ends_blank, t.seq_id)) + f.weight(e));
                    } else if (t.ends_blank && f.output(e) != blk) {
                        auto new_seq = old_id_seq[t.seq_id];
                        new_seq.push_back(f.output(e));
                        int s = reg(new_seq);
                        if (!ebt::in(std::make_pair(false, s), path_score)) {
                            heap.push_back(extra_data {f.head(e), false, s});
                        }
                        path_score[std::make_pair(false, s)]
                            = ebt::log_add(get_score(path_score, false, s),
                                old_path_score.at(std::make_pair(t.ends_blank, t.seq_id)) + f.weight(e));
                    } else if (!t.ends_blank && f.output(e) == blk) {
                        int s = reg(old_id_seq[t.seq_id]);
                        if (!ebt::in(std::make_pair(true, s), path_score)) {
                            heap.push_back(extra_data {f.head(e), true, s});
                        }
                        path_score[std::make_pair(true, s)]
                            = ebt::log_add(get_score(path_score, true, s),
                                old_path_score.at(std::make_pair(t.ends_blank, t.seq_id)) + f.weight(e));
                    } else if (!t.ends_blank && old_id_seq[t.seq_id].back() == f.output(e)) {
                        int s = reg(old_id_seq[t.seq_id]);
                        if (!ebt::in(std::make_pair(false, s), path_score)) {
                            heap.push_back(extra_data {f.head(e), false, s});
                        }
                        path_score[std::make_pair(false, s)]
                            = ebt::log_add(get_score(path_score, false, s),
                                old_path_score.at(std::make_pair(t.ends_blank, t.seq_id)) + f.weight(e));
                    } else if (!t.ends_blank && old_id_seq[t.seq_id].back() != f.output(e)) {
                        auto new_seq = old_id_seq[t.seq_id];
                        new_seq.push_back(f.output(e));
                        int s = reg(new_seq);
                        if (!ebt::in(std::make_pair(false, s), path_score)) {
                            heap.push_back(extra_data {f.head(e), false, s});
                        }
                        path_score[std::make_pair(false, s)]
                            = ebt::log_add(get_score(path_score, false, s),
                                old_path_score.at(std::make_pair(t.ends_blank, t.seq_id)) + f.weight(e));
                    }

                    std::push_heap(heap.begin(), heap.end(), less);
                    dirty = true;
                }
            }

            while (heap.size() > topk) {
                std::pop_heap(heap.begin(), heap.end(), less);
                heap.pop_back();
            }
        }
    }

}
