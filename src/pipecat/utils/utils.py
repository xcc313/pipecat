#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import collections
import itertools
from typing import Coroutine, Optional, Set

from loguru import logger

_COUNTS = collections.defaultdict(itertools.count)
_ID = itertools.count()
_TASKS: Set[asyncio.Task] = set()


def obj_id() -> int:
    """Generate a unique id for an object.

    >>> obj_id()
    0
    >>> obj_id()
    1
    >>> obj_id()
    2
    """
    return next(_ID)


def obj_count(obj) -> int:
    """Generate a unique id for an object.

    >>> obj_count(object())
    0
    >>> obj_count(object())
    1
    >>> new_type = type('NewType', (object,), {})
    >>> obj_count(new_type())
    0
    """
    return next(_COUNTS[obj.__class__.__name__])


def create_task(loop: asyncio.AbstractEventLoop, coroutine: Coroutine, name: str) -> asyncio.Task:
    async def run_coroutine():
        try:
            await coroutine
        except asyncio.CancelledError:
            logger.trace(f"{name}: task cancelled")
            # Re-raise the exception to ensure the task is cancelled.
            raise
        except Exception as e:
            logger.exception(f"{name}: unexpected exception: {e}")

    task = loop.create_task(run_coroutine())
    task.set_name(name)
    _TASKS.add(task)
    logger.trace(f"{name}: task created")
    return task


async def wait_for_task(task: asyncio.Task, timeout: Optional[float] = None):
    name = task.get_name()
    try:
        if timeout:
            await asyncio.wait_for(task, timeout=timeout)
        else:
            await task
    except asyncio.TimeoutError:
        logger.warning(f"{name}: timed out waiting for task to finish")
    except asyncio.CancelledError:
        logger.error(f"{name}: unexpected task cancellation")
    except Exception as e:
        logger.exception(f"{name}: unexpected exception while stopping task: {e}")
    finally:
        try:
            _TASKS.remove(task)
        except KeyError as e:
            logger.error(f"{name}: error removing task (already removed?): {e}")


async def cancel_task(task: asyncio.Task, timeout: Optional[float] = None):
    name = task.get_name()
    task.cancel()
    try:
        if timeout:
            await asyncio.wait_for(task, timeout=timeout)
        else:
            await task
    except asyncio.TimeoutError:
        logger.warning(f"{name}: timed out waiting for task to cancel")
    except asyncio.CancelledError:
        # Here are sure the task is cancelled properly.
        pass
    except Exception as e:
        logger.exception(f"{name}: unexpected exception while cancelling task: {e}")
    finally:
        try:
            _TASKS.remove(task)
        except KeyError as e:
            logger.error(f"{name}: error removing task (already removed?): {e}")


def current_tasks() -> Set[asyncio.Task]:
    return _TASKS
