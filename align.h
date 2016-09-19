#ifndef ALIGN_H
#define ALIGN_H

#include "seg/ilat.h"
#include "seg/iscrf.h"
#include "seg/pair_scrf.h"

namespace iscrf {

    ilat::fst make_label_seq_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    void make_alignment_gold(
        scrf::dense_vec const& ali_param,
        std::vector<std::string> const& label_seq,
        learning_sample& s,
        learning_args const& i_args);

    void make_even_gold(
        std::vector<std::string> const& label_seq,
        learning_sample& s,
        learning_args const& l_args);

}

#endif
