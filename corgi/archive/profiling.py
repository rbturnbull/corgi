import logging


def display_profiling():
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        print("Profiling:")
        print(f"System Time: {usage.ru_stime:.4g}s")
        print(f"User Time: {usage.ru_utime:.4g}s")
        print(f"Max RSS: {usage.ru_maxrss} (units dependent on system)")
    except:
        logging.error("Profiling information could not be displayed.")
