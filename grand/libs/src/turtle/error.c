/* Capture of TURTLE error messages */

static char * error_msg = NULL;

static void capture_error(enum turtle_return code,
    turtle_function_t * function, const char * message)
{
        free(error_msg);
        const size_t n = strlen(message) + 1;
        error_msg = malloc(n);
        if (error_msg != NULL)
                memcpy(error_msg, message, n);
}

void turtle_error_set_trap(void)
{
        turtle_error_handler_set(&capture_error);
}

const char * turtle_error_get_last(void)
{
        return error_msg;
}
