#ifndef PAIR_HASH_H
#define PAIR_HASH_H

#include <utility>    // For std::pair
#include <functional> // For std::hash
#include <cstddef>    // For std::size_t

// Custom hash function for std::pair<int, int>
struct pair_hash {
    std::size_t operator () (const std::pair<int, int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);

        // One approach for hasing an int pair
        h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1; // Simple combination
    }
};

#endif // PAIR_HASH_H
