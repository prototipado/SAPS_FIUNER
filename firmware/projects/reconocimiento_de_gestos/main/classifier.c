#include "Classifier.h"

/**
* Predict class for features vector
*/
int predict(float *x) {
    uint8_t votes[4] = { 0 };
    // tree #1
    if (x[9] <= 0.4579762667417526) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[6] <= 43.0) {
                votes[2] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #2
    if (x[9] <= 0.4579762667417526) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[3] <= 38.0) {
                votes[3] += 1;
            }

            else {
                votes[2] += 1;
            }
        }
    }

    // tree #3
    if (x[9] <= 0.4579762667417526) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[4] <= 43.5) {
                votes[2] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #4
    if (x[10] <= 0.3541656732559204) {
        votes[0] += 1;
    }

    else {
        if (x[2] <= -0.009695466607809067) {
            votes[2] += 1;
        }

        else {
            if (x[6] <= 43.0) {
                votes[1] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #5
    if (x[10] <= 0.3541656732559204) {
        votes[0] += 1;
    }

    else {
        if (x[10] <= 0.6777433753013611) {
            votes[2] += 1;
        }

        else {
            if (x[2] <= 0.01919714082032442) {
                votes[1] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #6
    if (x[9] <= 0.4579762667417526) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[2] <= 0.008872833102941513) {
                votes[2] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #7
    if (x[11] <= 0.2662547305226326) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[3] <= 38.0) {
                votes[3] += 1;
            }

            else {
                votes[2] += 1;
            }
        }
    }

    // tree #8
    if (x[10] <= 0.3541656732559204) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[10] <= 0.6777433753013611) {
                votes[2] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #9
    if (x[11] <= 0.2662547305226326) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[0] <= -0.001643456518650055) {
                votes[2] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // tree #10
    if (x[9] <= 0.4579762667417526) {
        votes[0] += 1;
    }

    else {
        if (x[11] <= 0.6880477517843246) {
            votes[1] += 1;
        }

        else {
            if (x[6] <= 43.0) {
                votes[2] += 1;
            }

            else {
                votes[3] += 1;
            }
        }
    }

    // return argmax of votes
    uint8_t classIdx = 0;
    float maxVotes = votes[0];

    for (uint8_t i = 1; i < 4; i++) {
        if (votes[i] > maxVotes) {
            classIdx = i;
            maxVotes = votes[i];
        }
    }

    return classIdx;
};
/**
* Predict readable class name
*/
const char* predictLabel(float *x) {
    return idxToLabel(predict(x));
};
/**
* Convert class idx to readable name
*/
const char* idxToLabel(uint8_t classIdx) {
    switch (classIdx) {
        case 0:
        return "espera";
        case 1:
        return "globo";
        case 2:
        return "reves";
        case 3:
        return "smash";
        default:
        return "Houston we have a problem";
    }
};