#include "gen_ncf_path.h"
void CreateDir(char *sPathName)
{
    char DirName[512];
    strcpy(DirName, sPathName);
    int i, len = strlen(DirName);
    for (i = 1; i < len; i++)
    {
        if (DirName[i] == '/')
        {
            DirName[i] = '\0';
            if (access(DirName, 0) != 0)
            {
                if (mkdir(DirName, 0755) == -1)
                {
                    // Check again if the directory exists after the error
                    if (access(DirName, 0) != 0)
                    {
                        printf("[INFO] Error creating %s. Permission denied\n", DirName);
                    }
                }
            }
            DirName[i] = '/';
        }
    }
    if (len > 0 && access(DirName, 0) != 0)
    {
        if (mkdir(DirName, 0755) == -1)
        {
            // Check again if the directory exists after the error
            if (access(DirName, 0) != 0)
            {
                printf("[INFO] Error creating %s. Permission denied\n", DirName);
            }
        }
    }
}

char *my_strdup(const char *s)
{
    size_t len = strlen(s) + 1;
    char *copy = malloc(len);
    if (copy)
    {
        memcpy(copy, s, len);
    }
    return copy;
}

void SplitFileName(const char *fname,
                   const char *delimiter,
                   char *net_pair_str,
                   char *sta_pair_str,
                   char *cmp_pair_str,
                   char *suffix)
{
    /* 参数检查 */
    if (!fname || !delimiter || !net_pair_str ||
        !sta_pair_str || !cmp_pair_str || !suffix)
    {
        return;
    }

    char *fname_copy = my_strdup(fname); /* 不修改原字符串 */
    if (!fname_copy)
        return;

    char *saveptr = NULL;
    char *token = strtok_r(fname_copy, delimiter, &saveptr); /* net_pair */
    if (!token)
        goto cleanup;
    strcpy(net_pair_str, token);

    token = strtok_r(NULL, delimiter, &saveptr); /* sta_pair */
    if (!token)
        goto cleanup;
    strcpy(sta_pair_str, token);

    token = strtok_r(NULL, delimiter, &saveptr); /* cmp_pair */
    if (!token)
        goto cleanup;
    strcpy(cmp_pair_str, token);

    token = strtok_r(NULL, delimiter, &saveptr); /* suffix    */
    if (!token)
        goto cleanup;
    strcpy(suffix, token);

cleanup:
    free(fname_copy);
}