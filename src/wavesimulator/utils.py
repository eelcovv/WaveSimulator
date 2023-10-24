import argparse
import datetime
import errno
import logging
import os
import re
import subprocess
import sys
from time import strptime
import pathlib
import dateutil.parser as dparser

import numpy as np
import pandas as pd

MSG_FORMAT = "{:30s} : {}"


def clear_argument_list(argv):
    """
    Small utility to remove the \'\\\\r\' character from the last argument of the argv list
    appearing in cygwin

    Parameters
    ----------
    argv : list
        The argument list stored in `sys.argv`

    Returns
    -------
    list
        Cleared argument list

    """
    new_argv = list()
    for arg in argv:
        # replace the '\r' character with a empty space
        arg = re.sub("\r", "", arg)
        if arg != "":
            # only add the argument if it is not empty
            new_argv.append(arg)
    return new_argv


def get_logger(name):
    """Get the logger of the current level and set the level based on the main routine. Then return
    it

    Parameters
    ----------
    name : str
        the name of the logger to set.

    Returns
    -------
    type
        log: a handle of the current logger

    Notes
    -----
    This routine is used on top of each function to get the handle to the current logger and
    automatically set the verbosity level of the logger based on the main function

    Examples
    --------

    Assume you define a function which need to generate logging information based on the logger
    created in the main program. In that case you can do

    >>> def small_function():
    ...    logger = get_logger(__name__)
    ...    logger.info("Inside 'small_function' This is information to the user")
    ...    logger.debug("Inside 'small_function' This is some debugging stuff")
    ...    logger.warning("Inside 'small_function' This is a warning")
    ...    logger.critical("Inside 'small_function' The world is collapsing!")

    The logger can be created in the main program using the create_logger routine

    >>> def main(logging_level):
    ...     main_logger = create_logger(console_log_level=logging_level)
    ...     main_logger.info("Some information in the main")
    ...     main_logger.debug("Now we are calling the function")
    ...     small_function()
    ...     main_logger.debug("We are back in the main function")

    Let's call the main fuction in DEBUGGING mode

    >>> main(logging.DEBUG)
      INFO : Some information in the main
     DEBUG : Now we are calling the function
      INFO : Inside 'small_function' This is information to the user
     DEBUG : Inside 'small_function' This is some debugging stuff
    WARNING : Inside 'small_function' This is a warning
    CRITICAL : Inside 'small_function' The world is collapsing!
     DEBUG : We are back in the main function


    You can see that the logging level inside the `small_function` is obtained from the main level.
    Do the same but now in the normal information mode

    >>> main(logging.INFO)
      INFO : Some information in the main
      INFO : Inside 'small_function' This is information to the user
    WARNING : Inside 'small_function' This is a warning
    CRITICAL : Inside 'small_function' The world is collapsing!

    We can call in the silent mode, suppressing all debugging and normal info, but not Warnings

    >>> main(logging.WARNING)
    WARNING : Inside 'small_function' This is a warning
    CRITICAL : Inside 'small_function' The world is collapsing!

    Finally, to suppress everything except for critical warnings

    >>> main(logging.CRITICAL)
    CRITICAL : Inside 'small_function' The world is collapsing!

    """
    # the logger is based on the current main routine
    log = logging.getLogger(name)
    log.setLevel(logging.getLogger("__main__").getEffectiveLevel())
    return log


def is_exe(fpath):
    """Test if a file is an executable

    Parameters
    ----------
    fpath : str
        return true or false:

    Returns
    -------
    bool
        In case `fpath` is a file that can be executed return True, else False

    Notes
    -----
    This function can only be used on Linux file systems as the `which` command is used to identity
    the location of the program.
    """
    # use system command 'which' to locate the full location of the file
    p = subprocess.Popen(
        "which {}".format(fpath),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    p_list = p.stdout.read().splitlines()
    if p_list:
        # which return a path so copy it to fpath
        fpath = p_list[0]
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def clear_path(path_name):
    """routine to clear spurious dots and slashes from a path name
     example bla/././oke becomes bla/oke

    Parameters
    ----------
    path_name :
        return: clear_path as a string

    Returns
    -------
    type
        clear_path as a string

    Examples
    --------

    >>> long_path = os.path.join(".", "..", "ok", "yoo", ".", ".", "") + "/"
    >>> print(long_path)
    .\..\ok\\yoo\.\.\/
    >>> print(clear_path(long_path))
    ..\\ok\\yoo

    """
    return str(pathlib.PurePath(path_name))


def create_logger(
    log_file=None,
    console_log_level=logging.INFO,
    console_log_format_long=False,
    console_log_format_clean=False,
    file_log_level=logging.INFO,
    file_log_format_long=True,
    redirect_stderr=True,
):
    """Create a console logger

    Parameters
    ----------
    log_file : str, optional
        The name of the log file in case we want to write it to file. If it is not specified, no
        file is created
    console_log_level: int, optional
        The level of the console output. Defaults to logging.INFO
    console_log_format_long : bool
        Use a long informative format for the logging output to the console
    console_log_format_clean : bool
        Use a very clean format for the logging output.  If given together with
        consosl_log_format_long an
        AssertionError is raised
    file_log_level: int, optional
        In case the log file is used, specify the log level. Can be different from the console log
        level. Defaults to logging.INFO
    file_log_format_long: bool, optional
        Use a longer format for the file output. Default to True
    redirect_stderr: bool, optional
        If True the stderr output is written to a file with .err extension in stated of .out.
        Default = True

    Returns
    -------
    object
        The handle to the logger which we can use to create output to the screen using the logging
        module

    Examples
    --------

    Create a logger at the verbosity level, so no debug information is generated

    >>> logger = create_logger()
    >>> logger.debug("This is a debug message")

    The info and warning message are both plotted

    >>> logger.info("This is a information message")
      INFO : This is a information message
    >>> logger.warning("This is a warning message")
    WARNING : This is a warning message

    Create a logger at the debug level

    >>> logger = create_logger(console_log_level=logging.DEBUG)
    >>> logger.debug("This is a debug message")
     DEBUG : This is a debug message
    >>> logger.info("This is a information message")
      INFO : This is a information message
    >>> logger.warning("This is a warning message")
    WARNING : This is a warning message

    Create a logger at the warning level. All output is suppressed, except for the warnings

    >>> logger = create_logger(console_log_level=logging.WARNING)
    >>> logger.debug("This is a debug message")
    >>> logger.info("This is a information message")
    >>> logger.warning("This is a warning message")
    WARNING : This is a warning message

    It is also possible to redirect the output to a file. The file name given without an extension,
    as two file are created: one with the extension .out and one with the extension .err, for the
    normal user generated out put and system errors output respectively.

    >>> data_dir = os.path.join(os.path.split(__file__)[0], "..", "data")
    >>> file_name = os.path.join(data_dir, "log_file")
    >>> logger = create_logger(log_file=file_name,  console_log_level=logging.INFO,
    ... file_log_level=logging.DEBUG, file_log_format_long=False)
    >>> logger.debug("This is a debug message")
    >>> logger.info("This is a information message")
      INFO : This is a information message
    >>> logger.warning("This is a warning message")
    WARNING : This is a warning message
    >>> print("system normal message")
    system normal message
    >>> print("system error message", file=sys.stderr)

    At this point, two files have been generated, log_file.out and log_file.err. The first contains
    the normal logging output whereas the second contains error message generated by other packages
    which do not use the logging module. Note that the normal print statement shows up in the
    console but not in the file, whereas the second print statement to the stderr output does not
    show on the screen but is written to log_file.err

    To show the contents of the generated files we do

    >>> with open(file_name+".out", "r") as fp:
    ...   for line in fp.readlines():
    ...       print(line.strip())
    DEBUG : This is a debug message
    INFO : This is a information message
    WARNING : This is a warning message
    >>> sys.stderr.flush()  # forces to flush the stderr output buffer to file
    >>> with open(file_name + ".err", "r") as fp:
    ...   for line in fp.readlines():
    ...       print(line.strip())
    system error message

    References
    ----------
    https://docs.python.org/3/library/logging.html#levels

    """

    # start with creating the logger with a DEBUG level
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    # create a console handle with a console log level which may be higher than the current level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(console_log_level)

    fh = None

    # create file handler if a file name is given with more info
    if log_file is not None:
        log_file_out = log_file + ".out"
        fh = logging.FileHandler(log_file_out, mode="w")
        fh.setLevel(file_log_level)

        if redirect_stderr:
            error_file = log_file + ".err"
            sys.stderr = open(error_file, "w")

    formatter_long = logging.Formatter(
        "[%(asctime)s] %(levelname)8s --- %(message)s " + "(%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter_normal = logging.Formatter("%(levelname)6s : %(message)s")

    formatter_short = logging.Formatter("%(message)s")

    if console_log_format_clean and console_log_format_long:
        raise AssertionError(
            "Can only specify either a long or a short logging format. Not both "
            "at the same time"
        )

    # create formatter and add it to the handlers for the console output
    if console_log_format_long:
        formatter_cons = formatter_long
    elif console_log_format_clean:
        formatter_cons = formatter_short
    else:
        formatter_cons = formatter_normal

    ch.setFormatter(formatter_cons)

    if log_file is not None:
        if file_log_format_long:
            formatter_file = formatter_long
        else:
            formatter_file = formatter_normal

        # create console handler with a higher log level
        fh.setFormatter(formatter_file)

        # add the handlers to the logger
        logger.addHandler(fh)

    logger.addHandler(ch)
    if log_file:
        logger.addHandler(fh)

    return logger


def get_parameter_list_key(parlist):
    """Utility for the qtgraphs Parameter Item

    Parameters
    ----------
    parlist :
        the parameter tree list (is an ordered list) contain all the values and current one

    Returns
    -------
    str
        The name belonging to the current value

    Notes
    -----
    The parameter tree widget has a field 'list' in which a list of values is given with
    corresponding integers. The current integer belonging to the parlist is obtained by
    parlist.value() however, to get the associated value of the key field is less straightforward
    In this routine it is retrieved
    """

    # the current value of the parlist
    value = parlist.value()

    # get the reverse list from the parlist
    reverselist = parlist.reverse

    # get the index belonging to the current value
    index = reverselist[0].index(value)

    # get the name of the key of this index
    keyname = reverselist[1][index]

    return keyname


def get_column_with_max_cumulative_value(data, regular_expression=".*"):
    """Find the column of a pandas DataFrame with the maximum cumulative value

    Parameters
    ----------
    data : DataFrame
        Data frame with the columns
    regular_expression : str
        Regular expression used to make a selection of columns to include. Default to '.*', which
        means that all columns are included

    Returns
    -------
    str or None:
        The name of the column with the maximum cumulative value or None if no columns were found

    Notes
    -----
    * Only the columns with a name obeying the regular expression are taken into account

    * An example of usage can be found in the fatigue monitoring software where we have data frames
      with damage over all the channels at a hot spots. If you want to obtained the channel with
      the maximum cumulative damage you can use this function

    Examples
    --------

    >>> import string
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> n_cols = 5
    >>> n_rows = 10

    Create a 10 x 5 data frame with random values with columns named as A, B, C, etc

    >>> data_frame = pd.DataFrame(np.random.random_sample((n_rows, n_cols)),
    ...                           columns=list(string.ascii_uppercase)[:n_cols])

    Obtain the name of the column with the maximum cumulative value

    >>> get_column_with_max_cumulative_value(data_frame)
    'D'

    Obtain the name of the column with the maximum cumulative value only including colums A, B and C

    >>> get_column_with_max_cumulative_value(data_frame, regular_expression="[ABC]")
    'C'

    """

    columns = list()
    for col in data.columns.values:
        if re.match(regular_expression, col):
            columns.append(col)
    if columns:
        name_of_maximum_column = data.fillna(0).sum()[columns[0] : columns[-1]].idxmax()
    else:
        name_of_maximum_column = None

    return name_of_maximum_column


def print_banner(
    title,
    top_symbol="-",
    bottom_symbol=None,
    side_symbol=None,
    width=80,
    to_stdout=False,
    no_top_and_bottom=False,
):
    """Create a banner for plotting a bigger title above each section in the log output

    Parameters
    ----------
    title :
        The title to plot
    top_symbol : str
        the symbol used for the top line. Default value = "-"
    bottom_symbol : str
        the symbol used for the bottom line. Assume same as top if None is given
        (Default value = None)
    side_symbol : str
        The side symbol. Assume same as top if None is given, except if top is -, then take |
        (Default value = None)
    width : int
        the width of the banner (Default value = 80)
    no_top_and_bottom : bool
        make a simple print without the top and bottom line (Default value = False)
    to_stdout : bool, optional
        Print the banner to the standard output of the console instead of the logging system.
        Defaults to False

    Examples
    --------

    >>> logger = create_logger(console_log_format_clean=True)
    >>> print_banner("This is the start of a section")
    <BLANKLINE>
    --------------------------------------------------------------------------------
    | This is the start of a section                                               |
    --------------------------------------------------------------------------------

    Notes
    -----

    Unless the option 'to_stdout' is set to True, the banner is printed via the logging system.
    Therefore, a logger needs to be created first using `create_logger`

    """
    logger = get_logger(__name__)

    logger.debug("message debug in print_banner")

    if bottom_symbol is None:
        bottom_symbol = top_symbol

    if side_symbol is None:
        if bool(re.match("-", top_symbol)):
            side_symbol = "|"
        else:
            side_symbol = top_symbol

    if not no_top_and_bottom:
        message_string = (
            "{}\n"
            + "{} ".format(side_symbol)
            + "{:"
            + "{:d}".format(width - 4)
            + "}"
            + " {}".format(side_symbol)
            + "\n{}"
        )
        message = message_string.format(
            top_symbol * width, title, bottom_symbol * width
        )
    else:
        message_string = (
            "{} ".format(side_symbol) + "{:" + "{:d}".format(width - 4) + "}"
        )
        message = message_string.format(title)
    if to_stdout:
        print("\n{}".format(message))
        sys.stdout.flush()
    else:
        logger.info("\n{}".format(message))


def move_script_path_to_back_of_search_path(script_file, append_at_the_end=True):
    """Move the name of a script to the front or the back of the search path

    Parameters
    ----------
    script_file : str
        Name of the script to move

    append_at_the_end: bool, optional, default=True
        Append the name of the script to the end. In case this flag is false, the script file is
        prepended to the path

    Returns
    -------
    list:
        The new system path stored in a list

    Notes
    -----
    This script is sometimes required if the __version string is messing up with
    another __version string

    Examples
    --------

    sys.path = move_script_path_to_back_of_search_path(__file__)
    """
    script_file = os.path.realpath(script_file)
    path_to_script = os.path.split(script_file)[0]
    path_to_script_forward = re.sub("\\\\", "/", path_to_script)
    new_sys_path = list()
    for path in sys.path:
        path_forward = re.sub("\\\\", "/", path)
        if path_forward != path_to_script_forward:
            new_sys_path.append(path)

    if append_at_the_end:
        new_sys_path.append(path_to_script)
    else:
        new_sys_path = [path_to_script] + new_sys_path
    return new_sys_path


def analyse_annotations(annotation):
    """Analyse the string `annotation` which compactly sets the properties of a label string such as
    position, size and color.

    Parameters
    ----------
    annotation : str
        label[@(xp, yp)[s10][a0][cRRGGBB]]

    Returns
    -------
    tuple of strings and integers
        text (str), x position (float) y position (float), color (str), size (int), axis (int)

    Notes
    -----
    * The annotation string can be just given a a label. This label optionally can be extended
      with a '@' sign to include more information, like the location (xp, yp), the size with s10,
      the color with c and the axis with a and a integer

    * The annotation format is mostly used as a compact way to provide information on the annotation
      via a user input file

    * When specifying a color with the c construct, make sure that you put the color at the end of
      the *annotation* string.

    * In the Parameter scription we used a hex formulation for the color like cFFAA00. However,
      also the color names like cblue or cred are allowed

    * In case you add chmc:red, the hmc definition of red will be used. Other hmc color
      definitions are:

         - blue
         - lightblue
         - red
         - lightred
         - green
         - darkcyan

    * Also we can add the xkcd_ color definitions as described in the matplotlib_ manual

    .. _xkcd:
        https://xkcd.com/color/rgb/

    .. _matplotlib:
        https://matplotlib.org/users/colors.html

    Examples
    --------

    Simple example of a label with all the properties set to default

    >>> analyse_annotations("simple_label")
    ('simple_label', 0.0, 0.0, 'black', 12, 0)

    place a label at position x=0.1, y=0.4

    >>> analyse_annotations("more_complex@0.1,0.4")
    ('more_complex', 0.1, 0.4, 'black', 12, 0)

    place a label at position x=0.8, y=0.9 with color red. Note that we need to add brackets
    around the location

    >>> analyse_annotations("colored_label@(0.8,0.9)cred")
    ('colored_label', 0.8, 0.9, 'red', 12, 0)

    place a label at position x=0.8, y=0.9 with color red. Note that we need to add brackets
    around the location

    >>> analyse_annotations("small_label@(0.8,0.9)s8")
    ('small_label', 0.8, 0.9, 'black', 8, 0)

    Place a label at position x=0.8, y=0.9 with color red and size 16. This time we need
    to add the color label at the end to extract it correcly

    >>> analyse_annotations("large_colored_label_in_axis_2@(0.8,0.9)s16a2cred")
    ('large_colored_label_in_axis_2', 0.8, 0.9, 'red', 16, 2)

    Finally lets show how you use more colours

    >>> analyse_annotations("label@(0.8,0.9)s16a2chmc:red")
    ('label', 0.8, 0.9, '#DD2E1A', 16, 2)

    As you can see, the hex code of HMC read is returned.

    To set the xkcd colors do

    >>> analyse_annotations("label@(0.8,0.9)s16a2cxkcd:red")
    ('label', 0.8, 0.9, 'xkcd:red', 16, 2)

    This color value 'xkcd:red' is understood by all matplotlib routines
    """
    log = get_logger(__name__)
    lx = 0.0
    ly = 0.0
    color = "black"
    axis = 0
    size = 12
    pos = "({},{})".format(lx, ly)
    # first replace all white spaces from the string, as it is not allowed
    try:
        # see if there is an @ sign indicating that the position is specified
        text, rest = annotation.split("@")
        try:
            # after the @ sign we start with the position between brackets (,). Find it
            pos, rest = re.sub("[(|)]", " ", rest).split()

            # now the rest is only a size s18 and a color cred or c#FFAA00 (hexa code). First get
            # the size, then the color
            size_pattern = "s(\d+)"
            axis_pattern = "a(\d)"
            color_pattern = "c(.*)"
            m = re.search(size_pattern, rest)
            if bool(m):
                size = int(m.group(1))
                rest = re.sub(size_pattern, "", rest)
            m = re.match(axis_pattern, rest)
            if bool(m):
                try:
                    axis = int(m.group(1))
                except ValueError:
                    log.warning("axis must by integers. Set zero")
                rest = re.sub(axis_pattern, "", rest)
            m = re.match(color_pattern, rest)
            if bool(m):
                color = m.group(1)

        except ValueError:
            # in case of a value error we did not have a rest, so try again without split
            pos = re.sub("[(|)]", " ", rest)
        finally:
            lx, ly = pos.split(",")
            lx = float(lx)
            ly = float(ly)
    except ValueError:
        # there as no @ sign: just return the text value with all the rest the defaults
        text = annotation

    if re.match("^hmc:", color):
        hmc_color_name = color.split(":")[1]
        try:
            color = c_hmc[hmc_color_name]
        except KeyError:
            log.warning("color name not recognised: {}. Keeping black".format(color))
            color = "black"

    return text, lx, ly, color, size, axis


def clean_up_artists(axis, artist_list):
    """Remove all the artists stored in the `artist_list` belonging to the `axis`.


    Parameters
    ----------
    axis : :class:matplotlib.axes.Axes
        Clean Artists (ie. items added to a matplotlib plot) belonging to this axis
    artist_list : list
        List of artist to remove.

    Notes
    -----

    In case of animation of complex plots with contours and labels (such as a timer) we sometimes
    need to take care of removing all the Artists which are changing every time step.
    This function takes care of that. It also also ensured that we are not running out of memory
    when too many Artists are added

    Examples
    --------

    >>> from matplotlib.pyplot import subplots
    >>> from numpy.random import random_sample

    Create a list which we use to store all the artists which need to be cleaned

    >>> artist_list = list()

    Create a plot of some random data

    >>> fig, ax = subplots(ncols=1, nrows=1)
    >>> data = random_sample((20, 30))
    >>> cs = ax.contourf(data)

    Store the contour Artist in a list

    >>> artist_list.append(cs)

    Now clean it again

    >>> clean_up_artists(ax, artist_list)

    """

    for artist in artist_list:
        try:
            # fist attempt: try to remove collection of contours for instance
            while artist.collections:
                for col in artist.collections:
                    artist.collections.remove(col)
                    try:
                        axis.collections.remove(col)
                    except ValueError:
                        pass

                artist.collections = []
                axis.collections = []
        except AttributeError:
            pass

        # second attempt, try to remove the text
        try:
            artist.remove()
        except (AttributeError, ValueError):
            pass


def clean_up_plot(artist_list):
    """A small script to clean up all lines or other items of a matplot lib plot.


    Parameters
    ----------
    artist_list :
        a list of items to clean up

    Notes
    -----

    Necessary if you want to loop over multiple plot and maintain the axes and only update the data.
    Basically this does the same as `clean_up_artists`

    """
    n_cleaned = 0
    log = get_logger(__name__)
    for ln in artist_list:
        n_cleaned += 1
        try:
            ln.pop(0).remove()
        except (IndexError, AttributeError):
            try:
                ln.remove()
            except (ValueError, TypeError, AttributeError):
                del ln
        else:
            n_cleaned -= 1
            log.debug("All clean up failed. ")
    artist_list = []
    return artist_list


def valid_date(s):
    """Check if supplied data *s* is a valid date for the format Year-Month-Day

    Parameters
    ----------
    s : str
        A valid date in the form of YYYY-MM-DD, so first the year, then the month, then the day

    Returns
    -------
    :class:`datetime`
        Date object with with the  year, month, day obtained from the valid string representation

    Raises
    ------
    argparse.ArgumentTypeError:

    Notes
    -----
    This is a helper function for the argument parser module `argparse` which allows you to check
    if the argument passed on the command line is a valid date.

    Examples
    --------

    This is the direct usage of `valid_date` to see if the date supplied is of format YYYY-MM-DD

    >>> try:
    ...     date = valid_date("1973-11-12")
    ... except argparse.ArgumentTypeError:
    ...     print("This date is invalid")
    ... else:
    ...     print("This date is valid")
    This date is valid

    In case an invalid date is supplied

    >>> try:
    ...     date = valid_date("1973-15-12")
    ... except argparse.ArgumentTypeError:
    ...     print("This date is invalid")
    ... else:
    ...     print("This date is valid")
    This date is invalid


    Here it is demonstrated how to add a '--startdate' command line option to the argparse parser
    which checks if a valid date is supplied

    >>> parser = argparse.ArgumentParser()
    >>> p = parser.add_argument("--startdate",
    ...                         help="The Start Date - format YYYY-MM-DD ",
    ...                         required=True,
    ...                         type=valid_date)

    References
    ----------

    https://stackoverflow.com/questions/25470844/specify-format-for-input-arguments-argparse-python
    """

    try:
        return strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.\nSupply date as YYYY-MM-DD".format(s)
        raise argparse.ArgumentTypeError(msg)


def get_path_depth(path_name):
    """
    Get the depth of a path or file name

    Parameters
    ----------
    path_name : str
        Path name to get the depth from

    Returns
    -------
    int
        depth of the path


    Examples
    --------

    >>> get_path_depth("C:\Anaconda")
    1
    >>> get_path_depth("C:\Anaconda\share")
    2
    >>> get_path_depth("C:\Anaconda\share\pywafo")
    3
    >>> get_path_depth(".\imaginary\path\subdir\share")
    4
    """

    if os.path.isfile(path_name) and os.path.exists(path_name):
        current_path = os.path.split(path_name)[0]
    else:
        current_path = path_name
    depth = 0
    previous_path = current_path
    while current_path not in ("", "."):
        current_path = os.path.split(current_path)[0]
        if current_path == previous_path:
            # for a full path name we end at the root 'C:\'. Detect that be comparing with the
            # previous round
            break
        previous_path = current_path
        depth += 1

    return depth


def scan_base_directory(
    walk_dir=".",
    supplied_file_list=None,
    file_has_string_pattern="",
    file_has_not_string_pattern="",
    dir_has_string_pattern="",
    dir_has_not_string_pattern="",
    start_date_time=None,
    end_date_time=None,
    time_zone=None,
    time_stamp_year_first=True,
    time_stamp_day_first=False,
    extension=None,
    max_depth=None,
    sort_file_base_names=False,
):
    """Recursively scan the directory `walk_dir` and get all files underneath obeying the search
    strings and/or date/time ranges

    Parameters
    ----------
    walk_dir : str, optional
        The base directory to start the import. Default = "."
    supplied_file_list: list, optional
        In case walk dir is not given we can explicitly pass a file list to analyse. Default = None
    dir_has_string_pattern : str, optional
        Requires the directory name to have this pattern (Default value = ""). This selection is
        only made on the first directory level below the walk_dir
    dir_has_not_string_pattern : str, optional
        Requires the directory name NOT to have this pattern (Default value = ""). This selection is
        only made on the first directory level below the walk_dir
    file_has_string_pattern : str, optional
        Requires the file name to have this pattern (Default value = "", i.e. matches all)
    file_has_not_string_pattern : str, optional
        Requires the file name NOT to have this pattern (Default value = "")
    extension : str or None, optional
        Extension of the file to match. If None, also matches. Default = None
    max_depth : int, optional
        Sets a maximum depth to which the search is carried out. Default = None, which does not
        limit the search depth. For deep file structures setting a limit to the search depth speeds
        up the search.
    sort_file_base_names: bool, option
        If True, sort the resulting file list alphabetically based on the file base name.
        Default = False
    start_date_time: DateTime or None, optional
        If given, get the date time from the current file name and only add the files with a
        date/time equal or large the *start_date_time*. Default is None
    end_date_time: DateTime or None, optional
        If given, get the date time from the current file name and only add the files with a
        date/time smaller than the *end_date_time*. Default is None
    time_zone:str or None, optional
        If given add this time zone to the file stamp. The start and end time should also have a
        time zone
    time_stamp_year_first: bool, optional
        Passed to the datetime parser. If true, the year is first in the date/time string.
        Default = True
    time_stamp_day_first: bool, optional
        Passed to the datetime parser. If true, the day is first in the date/time string.
        Default = False

    Returns
    -------
    list
        All the  file names found below the input directory `walk_dir` obeying all the search
        strings

    Examples
    --------

    Find all the python files under the share directory in the Anaconda installation folder

    >>> scan_dir = "C:\\Anaconda\\share"
    >>> file_list = scan_base_directory(scan_dir, extension='.py')

    Find all the python files under the share directory in the Anaconda installation folder
    belonging to the pywafo directory

    >>> file_list = scan_base_directory(scan_dir, extension='.py', dir_has_string_pattern="wafo")

    Note that wafo matches on the directory 'pywafo', which is the first directory level below the
    scan directory. However, if we would match on '^wafo' the returned list would be empty as the
    directory has to *start* with wafo.

    In order to get all the files with "test" in the name with a directory depth smaller than 3 do

    >>> file_list = scan_base_directory(scan_dir, extension='.py', dir_has_string_pattern="wafo",
    ...                                 file_has_string_pattern="test", max_depth=3)


    Test the date/time boundaries. First create a file list from 28 sep 2017 00:00 to 5:00 with a
    hour interval and convert it to a string list

    >>> file_names = ["AMS_{}.mdf".format(dt.strftime("%y%m%dT%H%M%S")) for dt in
    ...    pd.date_range("20170928T000000", "20170928T030000", freq="30min")]
    >>> for file_name in file_names:
    ...     print(file_name)
    AMS_170928T000000.mdf
    AMS_170928T003000.mdf
    AMS_170928T010000.mdf
    AMS_170928T013000.mdf
    AMS_170928T020000.mdf
    AMS_170928T023000.mdf
    AMS_170928T030000.mdf

    Use the scan_base_directory to get the files within a specific date/time range

    >>> file_selection = scan_base_directory(supplied_file_list=file_names,
    ...  start_date_time="20170928T010000", end_date_time="20170928T023000")

    >>> for file_name in file_selection:
    ...     print(file_name)
    AMS_170928T010000.mdf
    AMS_170928T013000.mdf
    AMS_170928T020000.mdf

    Note that the selected range run from 1 am until 2 am; the end_date_time of 2.30 am is not
    included

    """

    log = get_logger(__name__)

    # get the regular expression for the has_pattern and has_not_pattern of the files and
    # directories
    file_has_string = get_regex_pattern(file_has_string_pattern)
    file_has_not_string = get_regex_pattern(file_has_not_string_pattern)
    dir_has_string = get_regex_pattern(dir_has_string_pattern)
    dir_has_not_string = get_regex_pattern(dir_has_not_string_pattern)
    log.debug(MSG_FORMAT.format("file_has_string", file_has_string))
    log.debug(MSG_FORMAT.format("file_has_not_string", file_has_not_string))
    log.debug(MSG_FORMAT.format("dir_has_string", dir_has_string))
    log.debug(MSG_FORMAT.format("dir_has_not_string", dir_has_not_string))

    # use os.walk to recursively walk over all the file and directories
    top_directory = True
    file_list = list()
    log.debug("Scanning directory {}".format(walk_dir))
    for root, subdirs, files in os.walk(walk_dir, topdown=True):
        if supplied_file_list is not None:
            root = "."
            subdirs[:] = list()
            files = supplied_file_list

        log.debug("root={}  sub={} files={}".format(root, subdirs, files))
        log.debug(MSG_FORMAT.format("root", root))
        log.debug(MSG_FORMAT.format("sub dirs", subdirs))
        log.debug(MSG_FORMAT.format("files", files))
        # get the relative path towards the top directory (walk_dir)
        relative_path = os.path.relpath(root, walk_dir)

        depth = get_path_depth(relative_path)

        if root == walk_dir:
            top_directory = True
        else:
            top_directory = False

        # base on the first directory list we are going to make selection of directories to
        # process
        if top_directory:
            include_dirs = list()
            for subdir in subdirs:
                add_dir = False
                if dir_has_string is None or bool(dir_has_string.search(subdir)):
                    add_dir = True
                if add_dir and dir_has_not_string is not None:
                    if bool(dir_has_not_string.search(subdir)):
                        add_dir = False
                if add_dir:
                    include_dirs.append(subdir)
                # overrule the subdirectory list of os.walk:
                # http://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
                log.debug("Overruling subdirs with {}".format(include_dirs))
                subdirs[:] = include_dirs

        for filename in files:
            (filebase, ext) = os.path.splitext(filename)
            if extension is None or extension == ext:
                add_file = False

                if file_has_string is None or bool(file_has_string.search(filebase)):
                    # if has_string is none, the search pattern was either empty or invalid (which
                    # happens during typing the regex in the edit_box). In this case, always add the
                    # file. If not none, filter on the regex, so only add the file if the search
                    # pattern is in the filename
                    add_file = True

                # do not add the file in case the has_not string edit has been set (!="") and if the
                # file contains the pattern
                if add_file and file_has_not_string is not None:
                    if bool(file_has_not_string.search(filebase)):
                        # in case we want to exclude the file, the has_not search pattern must be
                        # valid so may not be None
                        add_file = False

                if add_file and (
                    start_date_time is not None or end_date_time is not None
                ):
                    # we have supplied a start time or a end time. See if we can get a date time
                    # from the file name
                    file_time_stamp = get_time_stamp_from_string(
                        string_with_date_time=filebase,
                        yearfirst=time_stamp_year_first,
                        dayfirst=time_stamp_day_first,
                        timezone=time_zone,
                    )

                    if file_time_stamp is not None:
                        # we found a file time stamp. Compare it with the start time
                        if start_date_time is not None:
                            if isinstance(start_date_time, str):
                                # in case the start time was supplied as a string
                                start_date_time = get_time_stamp_from_string(
                                    string_with_date_time=start_date_time,
                                    yearfirst=time_stamp_year_first,
                                    dayfirst=time_stamp_day_first,
                                    timezone=time_zone,
                                )

                            if file_time_stamp < start_date_time:
                                # the file time stamp is smaller, so don't add it
                                add_file = False
                        # if a end time is supplied. Also compare it with the end time
                        if end_date_time is not None:
                            if isinstance(end_date_time, str):
                                end_date_time = get_time_stamp_from_string(
                                    string_with_date_time=end_date_time,
                                    yearfirst=time_stamp_year_first,
                                    dayfirst=time_stamp_day_first,
                                    timezone=time_zone,
                                )
                            if file_time_stamp >= end_date_time:
                                # the file time stamp is larger, so don't add it
                                add_file = False

                if dir_has_string is not None and top_directory:
                    # in case we have specified a directory name with a string search, exclude the
                    # top directory
                    add_file = False

                if max_depth is not None and depth > max_depth:
                    add_file = False

                # create the full base name file
                file_name_to_add = os.path.join(walk_dir, relative_path, filebase)

                # get the path to the stl relative to the selected scan directory
                if add_file:
                    log.debug("Adding file {}".format(filebase))
                    file_list.append(clear_path(file_name_to_add + ext))

    # sort on the file name. First split the file base from the path, because if the file are in
    # different directories, the first file is not necessarily the oldest
    if sort_file_base_names:
        df = pd.DataFrame(
            data=file_list,
            index=[os.path.split(f)[1] for f in file_list],
            columns=["file_list"],
        )
        df.sort_index(inplace=True)
        file_list = df.file_list.values

    return file_list


def make_directory(directory):
    """Create a directory in case it does not yet exist.

    Parameters
    ----------
    directory : str
        Name of the directory to create

    Notes
    -----
    This function is used to create directories without checking if it already exist. If the
    directory already exists, we can silently continue.

    Raises
    ------
    OSError
        The OSError is only raised if it is not an `EEXIST` error. This implies that the creation
        of the directory failed due to another reason than the directory already being present.
        It could be that the file system is full or that we may not have write permission

    """
    logger = get_logger(__name__)
    try:
        os.makedirs(directory)
        logger.debug("Created directory : {}".format(directory))
    except OSError as exc:
        # an OSError was raised, see what is the cause
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            # the output directory already exists, that is ok so just continue
            logger.debug(
                "Directory {} already exists. No problem, we just continue".format(
                    directory
                )
            )
        else:
            # something else was wrong. Raise an error
            logger.warning(
                "Failed to create the directory {} because raised:\n{}".format(
                    directory, exc
                )
            )
            raise


def get_regex_pattern(search_pattern):
    """Routine to turn a string into a regular expression which can be used to match a string

    Parameters
    ----------
    search_pattern : str
        A regular expression in the form of a string

    Returns
    -------
    None or compiled regular expression
        A regular expression as return by the re.compile fucntion or None in case a invalid regular
        expression was given

    Notes
    -----
    An empty string or an invalid search_pattern will yield a None return

    """
    regular_expresion = None
    if search_pattern != "":
        try:
            regular_expresion = re.compile("{}".format(search_pattern))
        except re.error:
            regular_expresion = None
    return regular_expresion


def clear_argument_list(argv):
    """
    Small utility to remove the \'\\\\r\' character from the last argument of the argv list
    appearing in cygwin

    Parameters
    ----------
    argv : list
        The argument list stored in `sys.argv`

    Returns
    -------
    list
        Cleared argument list

    """
    new_argv = list()
    for arg in argv:
        # replace the '\r' character with a empty space
        arg = re.sub("\r", "", arg)
        if arg != "":
            # only add the argument if it is not empty
            new_argv.append(arg)
    return new_argv


def query_yes_no(question, default_answer="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    Parameters
    ----------
    question : str
        A question to ask the user
    default_answer : str, optional
        A default answer that is given when only return is hit. Default to 'no'

    Returns
    -------
    str:
        "yes" or "no", depending on the input of the user
    """
    log = get_logger(__name__)
    valid = {"yes": "yes", "y": "yes", "ye": "yes", "no": "no", "n": "no"}
    if not default_answer:
        prompt = " [y/n] "
    elif default_answer == "yes":
        prompt = " [Y/n] "
    elif default_answer == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default_answer)

    while 1:
        # sys.stdout.write(question + prompt)
        log.warning(question + prompt)
        choice = input().lower()
        if default_answer is not None and choice == "":
            return default_answer
        elif choice in list(valid.keys()):
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def get_time_stamp_from_string(
    string_with_date_time, yearfirst=True, dayfirst=False, timezone=None
):
    """
    Try to get a date/time stamp from a string

    Parameters
    ----------
    string_with_date_time: str
        The string to analyses
    yearfirst: bool, optional
        if true put the year first. See *dateutils.parser*. Default = True
    dayfirst: bool, optional
        if true put the day first. See *dateutils.parser*. Default = False
    timezone: str or None, optional
        if given try to add this time zone:w

    Returns
    -------
    :obj:`DateTime`
        Pandas data time string

    Examples
    --------

    The  date time in the file 'AMSBALDER_160929T000000' is  29 sep 2016 and does not have a
    time zone specification. The returned time stamp does also not have a time zone

    >>> file_name="AMSBALDER_160929T000000"
    >>> time_stamp =get_time_stamp_from_string(string_with_date_time=file_name)
    >>> print("File name {} has time stamp {}".format(file_name, time_stamp))
    File name AMSBALDER_160929T000000 has time stamp 2016-09-29 00:00:00

    We can also force to add a time zone. The Etc/GMT-2 time zone is UTC + 2 time zone which is
    the central europe summer time (CEST) or the Europe/Amsterdam Summer time.

    >>> time_stamp =get_time_stamp_from_string(string_with_date_time=file_name,
    ...                                        timezone="Etc/GMT-2")
    >>> print("File name {} has time stamp {}".format(file_name, time_stamp))
    File name AMSBALDER_160929T000000 has time stamp 2016-09-29 00:00:00+02:00

    This time we assume the file name already contains a time zone, 2 hours + UTC. Since we
    already have a time zone, the *timezone* option can only convert the date time to the specified
    time zone.

    >>> file_name="AMSBALDER_160929T000000+02"
    >>> time_stamp =get_time_stamp_from_string(string_with_date_time=file_name,
    ...                                        timezone="Etc/GMT-2")
    >>> print("File name {} has time stamp {}".format(file_name, time_stamp))
    File name AMSBALDER_160929T000000+02 has time stamp 2016-09-29 00:00:00+02:00

    In case the time zone given by the *timezone* options differs with the time zone in the file
    name, the time zone is converted

    >>> file_name="AMSBALDER_160929T000000+00"
    >>> time_stamp =get_time_stamp_from_string(string_with_date_time=file_name,
    ...                                        timezone="Etc/GMT-2")
    >>> print("File name {} has time stamp {}".format(file_name, time_stamp))
    File name AMSBALDER_160929T000000+00 has time stamp 2016-09-29 02:00:00+02:00

    """
    try:
        file_time_stamp = dparser.parse(
            string_with_date_time, fuzzy=True, yearfirst=yearfirst, dayfirst=dayfirst
        )
        file_time_stamp = pd.Timestamp(file_time_stamp)
    except ValueError:
        file_time_stamp = None
    else:
        # we have found a time stamp. See if we have to add a time zone
        if timezone is not None:
            try:
                file_time_stamp = file_time_stamp.tz_localize(timezone)
            except TypeError:
                # a time zone was present already. Then try to convert it
                file_time_stamp = file_time_stamp.tz_convert(timezone)

    return file_time_stamp
