#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("usage: %s dump-file\n", argv[0]);
        return 0;
    }

    FILE *in = fopen(argv[0], "r");

    int nArgs;
    fread(&nArgs, sizeof(int), 1, in);
    printf("%d arguments", nArgs);

    fclose(in);

    return 0;
}
