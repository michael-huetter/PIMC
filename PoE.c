#include <stdlib.h>
#include <time.h>

void initialize_random_seed() {
    srand((unsigned int)time(NULL));
}

int count_overlap_terms(int *eState, int numTimeSlices) {
    int num_overlap_terms = 0;
    for (int i = 0; i < numTimeSlices; i++) {
        if (eState[i] != eState[(i + 1) % numTimeSlices]) {
            num_overlap_terms++;
        }
    }
    return num_overlap_terms;
}

// update eState
void performPoE(int i, int xi, int xi_change_interval, int *xi_possible, int numXi, int numTimeSlices, int n, int *eState, int *xi_current) {

    if (xi == 0) {
        int random_value = rand() % n; // Generate one random number between 0 and n-1
        for (int j = 0; j < numTimeSlices; j++) {
            eState[j] = random_value; // Set every element to the same random value
        }
    *xi_current = 0; // Set xi_current to 0 as specified
    } else if (xi == numTimeSlices && n == 2) {
        int p_i = rand() % numTimeSlices;
        eState[p_i] = rand() % n;
        for (int j = 1; j < numTimeSlices; j++) {
            eState[(p_i + j) % numTimeSlices] = (eState[p_i] + j) % 2;
        }
        *xi_current = count_overlap_terms(eState, numTimeSlices);
    } else {
        int p_i = rand() % numTimeSlices;
        eState[p_i] = rand() % n;
        *xi_current = count_overlap_terms(eState, numTimeSlices);
        int j = 1;
        while (*xi_current != xi) {
            eState[(p_i + j) % numTimeSlices] = rand() % n;
            *xi_current = count_overlap_terms(eState, numTimeSlices);
            j++;
        }
    }
}
