/* commands.h: subcommands of the athanor binary, dispatched by athanor.c.
 *
 * Each subcommand receives the arguments after its name (argv[0] is the
 * subcommand name) and an arena for everything it allocates. It returns the
 * process exit code: 0 success, 1 failure, 2 usage error.
 */
#ifndef ATH_TOOLS_COMMANDS_H
#define ATH_TOOLS_COMMANDS_H

#include "core/arena.h"

#include <stdio.h>

#define ATH_EXIT_OK      0
#define ATH_EXIT_FAILURE 1
#define ATH_EXIT_USAGE   2

int  ath_cmd_inspect(ath_arena *a, int argc, char **argv);
void ath_cmd_inspect_usage(FILE *out);
int  ath_cmd_chat(ath_arena *a, int argc, char **argv);
void ath_cmd_chat_usage(FILE *out);
int  ath_cmd_devices(ath_arena *a, int argc, char **argv);
void ath_cmd_devices_usage(FILE *out);

#endif /* ATH_TOOLS_COMMANDS_H */
