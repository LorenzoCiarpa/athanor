/* athanor.c: the athanor binary. Only the dispatch table and process setup
 * live here; every subcommand is a function in its own file under tools/. */
#include "commands.h"

#include <string.h>

typedef struct {
    const char *name;
    const char *summary;
    int       (*run)(ath_arena *a, int argc, char **argv);
    void      (*usage)(FILE *out);
} command;

static const command commands[] = {
    {"inspect", "describe the config and every tensor of a checkpoint", ath_cmd_inspect,
     ath_cmd_inspect_usage},
    {"chat", "chat with a model, on the GPU or the CPU", ath_cmd_chat, ath_cmd_chat_usage},
    {"devices", "list the GPUs this build can use", ath_cmd_devices, ath_cmd_devices_usage},
    {"bench", "measure prefill and decode speed (docs/BENCHMARKS.md)", ath_cmd_bench, ath_cmd_bench_usage},
};

static void usage(FILE *out)
{
    fprintf(out, "usage: athanor <command> [options]\n\ncommands:\n");
    for (size_t i = 0; i < ATH_ARRAY_LEN(commands); i++)
        fprintf(out, "  %-10s %s\n", commands[i].name, commands[i].summary);
    fprintf(out, "\nRun 'athanor <command> --help' for the options of a command.\n"
                 "Set ATH_LOG=error|warn|info|debug|trace to change log verbosity (stderr).\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(stderr);
        return ATH_EXIT_USAGE;
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "help") == 0) {
        if (argc > 2 && strcmp(argv[1], "help") == 0) {
            for (size_t i = 0; i < ATH_ARRAY_LEN(commands); i++) {
                if (strcmp(argv[2], commands[i].name) == 0) {
                    commands[i].usage(stdout);
                    return ATH_EXIT_OK;
                }
            }
        }
        usage(stdout);
        return ATH_EXIT_OK;
    }

    const command *cmd = NULL;
    for (size_t i = 0; i < ATH_ARRAY_LEN(commands); i++)
        if (strcmp(argv[1], commands[i].name) == 0)
            cmd = &commands[i];
    if (!cmd) {
        fprintf(stderr, "athanor: unknown command '%s'\n\n", argv[1]);
        usage(stderr);
        return ATH_EXIT_USAGE;
    }

    /* All memory of a run comes from one arena, including the stdout buffer
     * that the C library would otherwise malloc on first use. stdout is
     * closed before the arena is freed, which also reports write errors
     * (a full disk, a closed pipe) instead of losing them at exit. */
    ath_arena a;
    ath_arena_init(&a, 0);
    size_t bufsize = (size_t)1 << 16;
    char *buf = ath_arena_alloc(&a, bufsize, 64);
    if (buf)
        setvbuf(stdout, buf, _IOFBF, bufsize);

    int rc = cmd->run(&a, argc - 1, argv + 1);

    if (fclose(stdout) != 0 && rc == ATH_EXIT_OK) {
        perror("athanor: write error");
        rc = ATH_EXIT_FAILURE;
    }
    ath_arena_free(&a);
    return rc;
}
