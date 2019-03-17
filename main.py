"""Module for testing the strength of passwords.

The strength of a password is tested by doing a nearest neighbor search with
a list of ~100k known common passwords.
"""

import argparse
import getpass
import sys

from numba import jit
import numpy as np

@jit
def edit_distance(left, right):
    """Computes the Levenshtein (i.e. edit) distance between two strings."""
    similarities = np.zeros((len(left) + 1, len(right) + 1), dtype=np.int32)
    similarities[:, 0] = range(len(left) + 1)
    similarities[0, :] = range(len(right) + 1)

    for l in range(1, len(left) + 1):
        for r in range(1, len(right) + 1):
            sub_cost = 0 if left[l - 1] == right[r - 1] else 1
            similarities[l][r] = min(similarities[l - 1][r] + 1,
                                     similarities[l][r - 1] + 1,
                                     similarities[l - 1][r - 1] + sub_cost)
    return similarities[len(left), len(right)]


def find_matches(password, max_distance=3):
    """Finds matching pwned passwords to the given password.

    Two passwords are considered to be matching if their edit distance <= 
    max_distance.

    Args:
        password: The password for which to find matching pwned passwords.
        max_distance: The distance above which two passwords are not considered
            matching.
    Returns:
        The list of matching pwned passwords.
    """
    matching_passwords = []
    with open('./data/passwords.txt', 'r') as file_:
        while True:
            pwned_password = file_.readline()
            if not pwned_password:
                break
            pwned_password = pwned_password.strip()
            distance = edit_distance(password, pwned_password)
            if distance <= max_distance:
                matching_passwords.append(pwned_password)
    return matching_passwords

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--show-matching',
        action='store_true',
        help='whether to return the list of matching passwords')
    parser.add_argument(
        '--max-distance',
        type=int,
        default=3,
        help='the distance above which passwords are not matching')
    args = parser.parse_args()
    password = getpass.getpass()
    matching_passwords = find_matches(password, args.max_distance)

    print("Found %d matching passwords." % len(matching_passwords))
    if args.show_matching:
        print(matching_passwords)
