#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of math and trigonometry Excel functions.
"""
import math
import functools
import collections
import numpy as np
import schedula as sh
from decimal import Decimal, ROUND_HALF_UP
from . import (
    get_error, raise_errors, is_number, flatten, wrap_ufunc, wrap_func,
    replace_empty, Error, xfilter, wrap_impure_func, COMPILING, to_number,
    clean_values, Array, XlError, _text2num, text2num
)
import re
from ..errors import FoundError

# Regex pattern for finding wildcards in criteria strings
_re_condition = re.compile('(?<!~)[?*]')

# noinspection PyDictCreation
FUNCTIONS = {}
FUNCTIONS['ABS'] = wrap_ufunc(np.abs)
FUNCTIONS['ACOS'] = wrap_ufunc(np.arccos)
FUNCTIONS['ACOSH'] = wrap_ufunc(np.arccosh)
FUNCTIONS['_XLFN.ACOT'] = FUNCTIONS['ACOT'] = wrap_ufunc(
    lambda x: (np.arctan(np.divide(1, x)) + np.pi) % np.pi
)
FUNCTIONS['ACOTH'] = wrap_ufunc(lambda x: np.arctanh(np.divide(1, x)))
FUNCTIONS['_XLFN.ACOTH'] = FUNCTIONS['ACOTH']


def xarabic(text):
    num = (1000, 500, 100, 50, 10, 5, 1)
    it = map(num.__getitem__, map('MDCLXVI'.index, str(text).upper()[::-1]))
    res, add, p = 0, True, -1
    for v in it:
        if p != v:
            add = p < v
        p = v
        if add:
            res += v
        else:
            res -= v
    return res


FUNCTIONS['_XLFN.ARABIC'] = FUNCTIONS['ARABIC'] = wrap_ufunc(
    xarabic, input_parser=lambda *a: a,
    args_parser=lambda *a: (replace_empty(x, '') for x in a)
)

FUNCTIONS['ASIN'] = wrap_ufunc(np.arcsin)
FUNCTIONS['ASINH'] = wrap_ufunc(np.arcsinh)
FUNCTIONS['ATAN'] = wrap_ufunc(np.arctan)


def xarctan2(x, y):
    return x == y == 0 and Error.errors['#DIV/0!'] or np.arctan2(y, x)


FUNCTIONS['ATAN2'] = wrap_ufunc(xarctan2)
FUNCTIONS['ATANH'] = wrap_ufunc(np.arctanh)
FUNCTIONS['COS'] = wrap_ufunc(np.cos)
FUNCTIONS['COSH'] = wrap_ufunc(np.cosh)


def xcot(x, func=np.tan):
    x = func(x)
    return (1 / x) if x else Error.errors['#DIV/0!']


FUNCTIONS['COT'] = FUNCTIONS['_XLFN.COT'] = wrap_ufunc(xcot)
FUNCTIONS['COTH'] = FUNCTIONS['_XLFN.COTH'] = wrap_ufunc(
    functools.partial(xcot, func=np.tanh)
)
FUNCTIONS['CSC'] = FUNCTIONS['_XLFN.CSC'] = wrap_ufunc(
    functools.partial(xcot, func=np.sin)
)
FUNCTIONS['CSCH'] = FUNCTIONS['_XLFN.CSCH'] = wrap_ufunc(
    functools.partial(xcot, func=np.sinh)
)


def xceiling(num, sig, ceil=math.ceil, dfl=0):
    if sig == 0:
        return dfl
    elif sig < 0 < num:
        return np.nan
    return ceil(num / sig) * sig


FUNCTIONS['CEILING'] = wrap_ufunc(xceiling)


def xceiling_math(num, sig=None, mode=0, ceil=math.ceil):
    if sig == 0:
        return 0
    elif sig is None:
        x, sig = abs(num), 1
    else:
        sig = abs(sig)
        x = num / sig
    if mode and num < 0:
        return -ceil(abs(x)) * sig
    return ceil(x) * sig


FUNCTIONS['CEILING.MATH'] = wrap_ufunc(xceiling_math)
FUNCTIONS['_XLFN.CEILING.MATH'] = FUNCTIONS['CEILING.MATH']
FUNCTIONS['CEILING.PRECISE'] = FUNCTIONS['CEILING.MATH']
FUNCTIONS['_XLFN.CEILING.PRECISE'] = FUNCTIONS['CEILING.PRECISE']
FUNCTIONS['DEGREES'] = wrap_ufunc(np.degrees)


def xdecimal(text, radix):
    text, radix = str(text), int(radix)
    try:
        return int(text, radix)
    except ValueError:
        return Error.errors['#NUM!']


FUNCTIONS['_XLFN.DECIMAL'] = FUNCTIONS['DECIMAL'] = wrap_ufunc(
    xdecimal, input_parser=lambda *a: a
)


def xeven(x):
    v = math.ceil(abs(x) / 2.) * 2
    return -v if x < 0 else v


FUNCTIONS['EVEN'] = wrap_ufunc(xeven)
FUNCTIONS['EXP'] = wrap_ufunc(np.exp)


def xfact(number, fact=math.factorial, limit=0):
    return np.nan if number < limit else int(fact(int(number or 0)))


FUNCTIONS['FACT'] = wrap_ufunc(xfact)


def _factdouble(x):
    return np.multiply.reduce(np.arange(max(x, 1), 0, -2))


def xfactdouble(number):
    raise_errors(number)
    x = next(flatten(number, None))
    if x is sh.EMPTY:
        x = 0
    if isinstance(x, bool):
        return Error.errors['#VALUE!']
    with np.errstate(divide='ignore', invalid='ignore'):
        # noinspection PyTypeChecker
        x = xfact(float(x), _factdouble, -1)
    return (np.isnan(x) or np.isinf(x)) and Error.errors['#NUM!'] or x


FUNCTIONS['FACTDOUBLE'] = wrap_func(xfactdouble)
FUNCTIONS['FLOOR'] = wrap_ufunc(
    functools.partial(xceiling, ceil=math.floor, dfl=Error.errors['#DIV/0!'])
)
FUNCTIONS['_XLFN.FLOOR.MATH'] = FUNCTIONS['FLOOR.MATH'] = wrap_ufunc(
    functools.partial(xceiling_math, ceil=math.floor)
)
FUNCTIONS['FLOOR.PRECISE'] = FUNCTIONS['FLOOR.MATH']
FUNCTIONS['_XLFN.FLOOR.PRECISE'] = FUNCTIONS['FLOOR.MATH']


def _xgcd(func, args):
    raise_errors(args)
    args = list(flatten(args, None))
    if not all(map(is_number, args)):
        return Error.errors['#VALUE!']
    args = np.array(args)
    if ((0 <= args) & (args <= (2 ** 53 + 1))).all():
        return func(args.astype(int))
    return Error.errors['#NUM!']


def xgcd(*args):
    return _xgcd(np.gcd.reduce, args)


FUNCTIONS['GCD'] = wrap_func(xgcd)
FUNCTIONS['INT'] = wrap_ufunc(math.floor)
FUNCTIONS['ISO.CEILING'] = FUNCTIONS['CEILING.PRECISE']


def xlcm(*args):
    return _xgcd(np.lcm.reduce, args)


FUNCTIONS['LCM'] = wrap_func(xlcm)
FUNCTIONS['LOG10'] = wrap_ufunc(np.log10)
FUNCTIONS['LOG'] = wrap_ufunc(
    lambda x, base=10: np.log(x) / np.log(base) if base else np.nan
)
FUNCTIONS['LN'] = wrap_ufunc(np.log)


def xmdeterm(x, func=np.linalg.det):
    raise_errors(x)
    if x.shape[0] != x.shape[1]:
        return Error.errors['#VALUE!']
    x = np.reshape(
        tuple(flatten(x, lambda v: not isinstance(v, str) and is_number(v))),
        x.shape
    )
    try:
        return np.around(func(x.astype(float)), 15).view(Array)
    except np.linalg.LinAlgError:
        return Error.errors['#NUM!']


FUNCTIONS['MDETERM'] = wrap_func(xmdeterm)
FUNCTIONS['MINVERSE'] = wrap_func(
    functools.partial(xmdeterm, func=np.linalg.inv)
)


def xmmult(x, y):
    raise_errors(x, y)
    return np.dot(x, y).astype(float).view(Array)


FUNCTIONS['MMULT'] = wrap_func(xmmult)


def xmod(x, y):
    return y == 0 and Error.errors['#DIV/0!'] or np.mod(x, y)


FUNCTIONS['MOD'] = wrap_ufunc(xmod)


def xmround(*args):
    raise_errors(args)
    num, sig = tuple(flatten(map(replace_empty, args), None))
    if isinstance(num, bool) or isinstance(sig, bool):
        return Error.errors['#VALUE!']
    num, sig = float(num), float(sig)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = num < 0 < sig and np.nan or xceiling(num, sig, ceil=np.round)
    return (np.isnan(x) or np.isinf(x)) and Error.errors['#NUM!'] or x


FUNCTIONS['MROUND'] = wrap_func(xmround)


def return_func(res, inp):
    shape = np.asarray(inp).shape
    if len(shape) == 2:
        return np.asarray([
            v if isinstance(v, XlError) or v is 1.0 else v[0][0]
            for v in res.ravel()
        ], dtype=object).reshape(shape).view(Array)
    return res


def xmunit(x):
    if x > 1:
        return np.identity(x)
    elif x == 1:
        return 1.0
    return Error.errors['#VALUE!']


FUNCTIONS['_XLFN.MUNIT'] = FUNCTIONS['MUNIT'] = wrap_ufunc(
    xmunit, input_parser=lambda *a: map(int, a), return_func=return_func,
    check_nan=False
)


def xodd(x):
    v = math.ceil(abs(x)) // 2 * 2 + 1
    return -v if x < 0 else v


FUNCTIONS['ODD'] = wrap_ufunc(xodd)
FUNCTIONS['PI'] = lambda: math.pi


def xpower(number, power):
    if number == 0:
        if power == 0:
            return Error.errors['#NUM!']
        if power < 0:
            return Error.errors['#DIV/0!']
    return np.power(number, power)


FUNCTIONS['POWER'] = wrap_ufunc(xpower)
FUNCTIONS['RADIANS'] = wrap_ufunc(np.radians)
FUNCTIONS['RAND'] = {
    'extra_inputs': collections.OrderedDict([(COMPILING, False)]),
    'function': wrap_impure_func(wrap_func(np.random.rand))
}


def xrandbetween(bottom, top):
    if isinstance(bottom, bool) or isinstance(top, bool):
        return Error.errors['#VALUE!']
    dx = top - bottom
    if dx < 0:
        return Error.errors['#NUM!']

    return bottom + dx * np.random.rand()


FUNCTIONS['RANDBETWEEN'] = wrap_ufunc(
    xrandbetween, input_parser=lambda *a: a,
    check_error=lambda *a: get_error(*a[::-1])
)


def _xroman(form):
    form = int(form + 1)
    num, let = (1000, 500, 100, 50, 10, 5, 1), 'MDCLXVI'
    for i, (n, l) in enumerate(zip(num, let), 1):
        yield n, l
        y = []
        for v, k in zip(num[i:], let[i:]):
            v = n - v
            if v not in num:
                y.append((v, k + l))
                if len(y) == form:
                    break
        yield from y[::-1]


def xroman(num, form=0):
    num, form = int(num), not isinstance(form, bool) and int(form or 0) or 0
    if not (0 <= num < 4000 and 0 <= form <= 4):
        raise ValueError()

    result = ""
    for i, n in _xroman(form):
        if not num:
            break
        count = int(num / i)
        result += n * count
        num -= i * count
    return result


FUNCTIONS['ROMAN'] = wrap_ufunc(xroman, input_parser=lambda *a: a)


def round_up(x):
    return float(Decimal(x).quantize(0, rounding=ROUND_HALF_UP))


def xround(x, d, func=round_up):
    d = 10 ** int(d)
    v = func(abs(x * d)) / d
    return -v if x < 0 else v


FUNCTIONS['ROUND'] = wrap_ufunc(xround)
FUNCTIONS['ROUNDDOWN'] = wrap_ufunc(functools.partial(xround, func=math.floor))
FUNCTIONS['ROUNDUP'] = wrap_ufunc(functools.partial(xround, func=math.ceil))
FUNCTIONS['SEC'] = FUNCTIONS['_XLFN.SEC'] = wrap_ufunc(
    functools.partial(xcot, func=np.cos)
)
FUNCTIONS['SECH'] = FUNCTIONS['_XLFN.SECH'] = wrap_ufunc(
    functools.partial(xcot, func=np.cosh)
)
FUNCTIONS['SIGN'] = wrap_ufunc(np.sign)
FUNCTIONS['SIN'] = wrap_ufunc(np.sin)
FUNCTIONS['SINH'] = wrap_ufunc(np.sinh)


def xsumproduct(*args):
    # Check all arrays are the same length
    # Excel returns #VAlUE! error if they don't match
    raise_errors(args)
    inp = np.asarray(args).reshape((len(args), -1))
    inp = inp[:, ~(inp == np.array(sh.EMPTY, dtype=object)).any(axis=0)]
    return np.sum(np.prod(np.nan_to_num(to_number(inp).astype(float)), axis=0))


FUNCTIONS['SUMPRODUCT'] = wrap_func(xsumproduct)
FUNCTIONS['SQRT'] = wrap_ufunc(np.sqrt)


def xsrqtpi(number):
    raise_errors(number)
    x = list(flatten(replace_empty(number), None))[0]
    if isinstance(x, bool):
        return Error.errors['#VALUE!']
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.sqrt(float(x) * np.pi)
    return (np.isnan(x) or np.isinf(x)) and Error.errors['#NUM!'] or x


FUNCTIONS['SQRTPI'] = wrap_func(xsrqtpi)


def xsum(*args, func=np.sum):
    raise_errors(args)
    inp = []
    for a in args:
        if isinstance(a, str) and not is_number(a):
            raise ValueError
        elif isinstance(a, bool):
            a = float(a)
        inp.append(np.asarray(a).reshape(1, -1))
    inp = to_number(clean_values(np.concatenate(inp, 1))).astype(float).ravel()
    return func(inp[~np.isnan(inp)])


FUNCTIONS['PRODUCT'] = wrap_func(functools.partial(xsum, func=np.prod))
FUNCTIONS['SUM'] = wrap_func(xsum)
FUNCTIONS['SUMIF'] = wrap_func(functools.partial(xfilter, xsum))

def xsumifs(sum_range, *args):
    """Sums values that meet multiple criteria.
    
    Args:
        sum_range: Range containing the values to sum.
        *args: Pairs of (criteria_range, criteria) to filter by.
    
    Returns:
        Sum of filtered values or an error.
    """
    # SUMIFS adds the cells specified by a given set of conditions or criteria.
    # The syntax of the SUMIFS function is:
    # SUMIFS(sum_range, criteria_range1, criteria1, [criteria_range2, criteria2], ...)
    # - sum_range is the range to sum
    # - criteria_range1 is the range to evaluate with criteria1
    # - criteria1 is the condition to apply to criteria_range1
    # - criteria_range2, criteria2, etc. are additional ranges and criteria
    raise_errors(sum_range, *args)
    
    if len(args) < 2 or len(args) % 2 != 0:
        return Error.errors['#VALUE!']
    
    # Convert to numpy arrays
    sum_range = np.asarray(sum_range, dtype=object)
    sum_range_shape = sum_range.shape
    
    # Get all criteria ranges and their criteria
    criteria_ranges = []
    criteria_values = []
    
    for i in range(0, len(args), 2):
        criteria_range = np.asarray(args[i], dtype=object)
        criteria = args[i+1]

        # if criteria is not a 1x1 array of arrays, then raise an error
        if not isinstance(criteria, np.ndarray) or criteria.shape != (1, 1):
            raise Exception("FAILURE in SUMIFS")
        
        criteria = criteria[0][0]
        
        # Check if ranges have compatible shapes
        if criteria_range.shape != sum_range_shape:
            return Error.errors['#VALUE!']
        
        criteria_ranges.append(criteria_range)
        criteria_values.append(criteria)
    
    # Create a mask where all criteria are met
    mask = np.ones(sum_range.shape, dtype=bool)
    
    for criteria_range, criteria in zip(criteria_ranges, criteria_values):

        
        
        # Apply xfilter logic for each criteria pair
        test_range = {'raw': replace_empty(criteria_range, '')}
        
        # Extract filtering logic similar to xfilter
        from .operators import LOGIC_OPERATORS
        operator = '='
        criteria_value = criteria
        
        if isinstance(criteria, str):
            # Handle wildcards similar to xfilter
            it = _re_condition.findall(criteria)
            if it:
                _ = lambda v: re.escape(v.replace('~?', '?').replace('~*', '*'))
                match = re.compile(''.join(sum(zip(
                    map(_, _re_condition.split(criteria)),
                    tuple(map(lambda v: '.%s' % v, it)) + ('',)
                ), ()))).match
                f = lambda v: isinstance(v, str) and bool(match(v))
                current_mask = np.vectorize(f, otypes=[bool])(test_range['raw'])
                mask = mask & current_mask
                continue
            elif any(v in criteria for v in ('~?', '~*')):
                criteria = criteria.replace('~?', '?').replace('~*', '*')

            # Extract operator from the beginning of criteria string
            for k in LOGIC_OPERATORS:
                if criteria.startswith(k) and criteria != k:
                    operator, criteria_value = k, criteria[len(k):]
                    break
            
            # Try to convert criteria to a numeric value if possible
            from ..tokens.operand import Number, Error
            from ..errors import TokenError
            
            try:
                # First check if it's a simple number
                for token in (Number, Error):
                    try:
                        token_obj = token(criteria_value)
                        if token_obj.end_match == len(criteria_value):
                            criteria_value = token_obj.compile()
                            break
                    except TokenError:
                        pass
            except Exception:
                # If conversion fails, keep original criteria
                pass
            
            # Convert text to number if possible
            if isinstance(criteria_value, str):
                criteria_value = _text2num(criteria_value)
        
        from .operators import _get_type_id
        type_id, operator_func = _get_type_id(criteria_value), LOGIC_OPERATORS[operator]
        
        # Create a function to check each value against criteria
        @functools.lru_cache()
        def check(value):
            try:
                # Handle type conversions for comparison
                if isinstance(value, str) and is_number(value) and is_number(criteria_value):
                    value = float(value)
                elif isinstance(criteria_value, str) and is_number(criteria_value) and is_number(value):
                    criteria_val = float(criteria_value)
                    return operator_func(value, criteria_val)
                
                # Check if types match before comparing
                if _get_type_id(value) == type_id:
                    return operator_func(value, criteria_value)
                # Special case for numeric comparisons where types might differ
                elif is_number(value) and is_number(criteria_value):
                    return operator_func(float(value), float(criteria_value))
                return False
            except:
                return False
        
        # Apply the check to create a boolean mask
        if is_number(criteria_value):
            test_range['num'] = text2num(test_range['raw'])
            values = test_range['num']
        else:
            values = test_range['raw']
        
        current_mask = np.vectorize(check, otypes=[bool])(values)
        mask = mask & current_mask
    
    # Apply the mask to sum_range and calculate sum
    try:
        filtered_values = sum_range[mask]
        filtered_values = to_number(filtered_values).astype(float)
        return np.sum(filtered_values[~np.isnan(filtered_values)])
    except FoundError as ex:
        return ex.err
    except Exception:
        return Error.errors['#VALUE!']

FUNCTIONS['SUMIFS'] = wrap_func(xsumifs)

FUNCTIONS['SUMSQ'] = wrap_func(functools.partial(
    xsum, func=lambda v: np.sum(np.square(v))
))
FUNCTIONS['TAN'] = wrap_ufunc(np.tan)
FUNCTIONS['TANH'] = wrap_ufunc(np.tanh)


def xtrunc(x, d=0, func=math.trunc):
    return xround(x, d=d, func=func)


FUNCTIONS['TRUNC'] = wrap_ufunc(xtrunc)
