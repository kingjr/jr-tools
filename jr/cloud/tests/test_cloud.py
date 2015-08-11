import os
import os.path as op
import numpy as np
from nose.tools import assert_raises, assert_equal, assert_true
from ..base import Client


def test_cloud():
    assert_raises(ValueError, Client, 'foo', 'a')

    for server in ['Dropbox', 'S3']:
        route = Client(server, bucket='jrking.test')
        # check clean
        for fname in ['foo.npy', 'foo2.npy', 'bar.npy']:
            route.delete(fname)

        # check upload
        # assert_raises(ValueError, route.upload, 'foo.npy', 'foo.npy')
        np.save('foo.npy', np.ones(1000))
        np.save('bar.npy', np.ones(2000))
        assert_true(route.upload('foo.npy', 'foo.npy'))
        assert_equal(op.getsize('foo.npy'), route.metadata('foo.npy')['bytes'])
        assert_true(not route.upload('foo.npy', 'foo.npy', overwrite=False))
        # skips if identical sizes?
        assert_true(not route.upload('foo.npy', 'foo.npy', overwrite='auto'))
        assert_true(route.upload('foo.npy', 'foo.npy', overwrite=True))
        assert_true(route.upload('bar.npy', 'foo.npy', overwrite='auto'))
        # clean
        assert_true(route.delete('foo.npy'))
        assert_true(not route.delete('bar.npy'))

        # check download
        np.save('foo.npy', np.ones(1000))
        np.save('bar.npy', np.ones(2000))
        route.upload('foo.npy', 'foo.npy')
        route.upload('bar.npy', 'bar.npy')
        try:
            os.remove('foo2.npy')
        except OSError:
            pass
        assert_true(route.download('foo.npy', 'foo2.npy'))
        assert_equal(op.getsize('foo2.npy'),
                     route.metadata('foo.npy')['bytes'])
        assert_true(not route.download('foo.npy', 'foo2.npy', overwrite=False))
        # skips if identical sizes?
        assert_true(not route.download('foo.npy', 'foo2.npy',
                                       overwrite='auto'))
        assert_true(route.download('foo.npy', 'bar.npy', overwrite=True))
        assert_true(route.download('bar.npy', 'bar.npy', overwrite='auto'))
        # clean
        for fname in ['foo.npy', 'foo2.npy', 'bar.npy']:
            route.delete(fname)
            os.remove(fname)

    # TODO: tests credentials with dictionaries
