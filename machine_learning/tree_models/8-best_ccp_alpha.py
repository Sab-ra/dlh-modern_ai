#!/usr/bin/env python3
"""
Find best ccp_alpha
"""


def get_best_alpha(
        clfs,
        train_scores,
        test_scores,
        ccp_alphas
):
    """
    Return the best ccp_alpha and its classifier
    """

    best_index = 0

    for i in range(1, len(clfs)):
        if test_scores[i] > test_scores[best_index]:
            best_index = i
        elif test_scores[i] == test_scores[best_index]:
            current_gap = (
                abs(train_scores[i] - test_scores[i])
            )
            best_gap = (
                abs(train_scores[best_index] - test_scores[best_index])
            )

            if current_gap < best_gap:
                best_index = i
            elif current_gap == best_gap:
                if ccp_alphas[i] > ccp_alphas[best_index]:
                    best_index = i

    return ccp_alphas[best_index], clfs[best_index]
