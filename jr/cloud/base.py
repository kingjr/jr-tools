import os
import os.path as op
import warnings
from mne.utils import ProgressBar


class Client():
    def __init__(self, server, credentials=None, bucket=None,
                 client_root='./', overwrite='auto', remove_on_upload=False,
                 multithread=True, offline=False):
        from multiprocessing.pool import ThreadPool
        # Default params
        self.client_root = client_root
        self.overwrite = overwrite
        self.remove_on_upload = remove_on_upload
        self.multithread = multithread
        self.offline = offline
        pool = ThreadPool(processes=1)
        self._async_result = pool.apply_async(self._connect,
                                              (server, credentials, bucket))

    def _connect(self, server, credentials, bucket):
        # Setup server
        route = None
        if server == 'S3':
            route = S3_client(credentials, bucket)
        elif server == 'Dropbox':
            route = Dropbox_client(credentials, bucket)
        elif server == 'offline':
            route = BaseClient(credentials, bucket)
        return route

    def _is_offline(self):
        if self.offline:
            warnings.warn('offline')
            print('offline!')
            return True
        else:
            # if online mode, wait until init is finalized
            if not hasattr(self, 'route'):
                print('Checking connection initialization...')
                self.route = self._async_result.get()
                if self.route is None:
                    raise ValueError('Unknown server')
            return False

    def _strip_client_root(self, f_server):
        """remove the client root path from the server file"""
        if self.client_root is None:
            return f_server
        if f_server[:len(self.client_root)] == self.client_root:
            f_server = f_server[len(self.client_root):]
        if f_server[0] == '/':
            f_server = f_server[1:]
        return f_server

    def download(self, f_server, f_client=None, overwrite=None):
        if self._is_offline():
            return
        # get default overwrite parameter
        if overwrite is None:
            overwrite = self.overwrite
        # default filenames
        f_server = self._strip_client_root(f_server)
        if f_client is None:
            f_client = op.join(self.client_root, f_server)
        # connect
        self.route.connect()
        # check that file exists online
        meta_data = self.route.metadata(f_server)
        # don't download if files are identical
        client_path = '/'.join(f_client.split('/')[:-1])
        if op.exists(f_client):
            if overwrite is False:
                print('%s already exists & no overwrite: skip' % f_client)
                return False
            elif overwrite == 'auto':
                fsize_client = op.getsize(f_client)
                if meta_data['bytes'] == fsize_client:
                    print('Identical files, download skipped %s' % f_client)
                    return False
        # create directory if doesn't exist already
        if ((not op.exists(client_path)) and (client_path is not None) and
                (client_path != '')):
            os.makedirs(client_path)
        # download
        self.route.download(f_server, f_client)
        print('Downloaded: %s > %s' % (f_server, f_client))
        return True

    def upload(self, f_client, f_server=None, overwrite=None,
               multithread=None, remove_on_upload=None):
        import threading
        if self._is_offline():
            return

        # get default params
        if multithread is None:
            multithread = self.multithread
        if overwrite is None:
            overwrite = self.overwrite
        if remove_on_upload is None:
            remove_on_upload = self.remove_on_upload

        if multithread:
            thread = threading.Thread(target=self._upload_thread,
                                      args=(f_client, f_server, overwrite,
                                            remove_on_upload))
            thread.start()
        else:
            return self._upload_thread(f_client, f_server, overwrite,
                                       remove_on_upload)

    def _upload_thread(self, f_client, f_server, overwrite, remove_on_upload):
        if f_server is None:
            f_server = f_client.split('/')[-1]
        f_server = self._strip_client_root(f_server)
        if op.isfile(f_client):
            return self._upload_file(f_client, f_server, overwrite,
                                     remove_on_upload)
        elif op.isdir(f_client):
            results = list()
            for root, dirs, files in os.walk(f_client):
                for filename in files:
                    # construct the full local path
                    local_path = op.join(root, filename)
                    # upload the file
                    results.append(self._upload_file(
                        local_path,
                        f_server + local_path.split(f_client)[-1],
                        overwrite, remove_on_upload))
            return sum(results)
        else:
            raise ValueError('File not found %s' % f_client)

    def _upload_file(self, f_client, f_server, overwrite, remove_on_upload):
        # connect
        self.route.connect()
        # check whether file exists online
        if overwrite is not True:
            metadata = self.route.metadata(f_server)
            # don't upload if files are identical
            if metadata['exist']:
                if overwrite is False:
                    print('File already exists & overwrite is False: skipped')
                    return False
                elif overwrite == 'auto':
                    fsize_client = op.getsize(f_client)
                    if metadata['bytes'] == fsize_client:
                        print('Identical files, upload skipped %s' % f_server)
                        return False
                else:
                    raise ValueError('overwrite must be bool or `auto`')
        # upload
        print('Uploading %s > %s' % (f_client,
                                     op.join(self.route.bucket, f_server)))
        self.route.upload(f_client, f_server)

        # remove_on_upload
        if remove_on_upload:
            os.remove(f_client)
        return True

    def metadata(self, f_server):
        if self._is_offline():
            return
        f_server = self._strip_client_root(f_server)
        return self.route.metadata(f_server)

    def delete(self, f_server):
        if self._is_offline():
            return
        f_server = self._strip_client_root(f_server)
        self.route.connect()
        return self.route.delete(f_server)


class BaseClient():
    def __init__(self, credentials=None, bucket=None):
        self.credentials = credentials
        self.bucket = bucket
        self.client = self.connect()

    def connect(self):
        return None

    def metadata(self, f_server):
        return None

    def download(self, f_server, f_client):
        pass

    def upload(self, f_client, f_server):
        pass

    def delete(self, f_server):
        return None


class S3_client():

    def __init__(self, credentials=None, bucket=None):
        if credentials is None:
            credentials = op.expanduser('~/.credentials/boto.cfg')
        self.credentials = credentials
        self.bucket = bucket
        self.client = self.connect()

    def connect(self):
        import boto
        if isinstance(self.credentials, str):
            with open(self.credentials, 'rb') as f:
                auth = f.read()
                auth = auth.split('\n')
                AWSAccessKeyId = auth[0].split('AWSAccessKeyId=')[1]
                AWSSecretKey = auth[1].split('AWSSecretKey=')[1]
        elif isinstance(self.credentials, dict):
            AWSAccessKeyId = self.credentials['AWSAccessKeyId']
            AWSSecretKey = self.credentials['AWSSecretKey']
        calling_format = 'boto.s3.connection.OrdinaryCallingFormat'
        client_ = boto.connect_s3(AWSAccessKeyId, AWSSecretKey,
                                  calling_format=calling_format)
        return client_.get_bucket(self.bucket)

    def metadata(self, f_server):
        key = self.client.get_key(f_server)
        if key is None:
            metadata = dict(exist=False, bytes=0)
        else:
            metadata = dict(exist=True, bytes=key.size)
        return metadata

    def download(self, f_server, f_client):
        key = self.client.get_key(f_server)
        key.get_contents_to_filename(f_client)

    def upload(self, f_client, f_server):
        key = self.client.get_key(f_server)
        if key is None:
            key = self.client.new_key(f_server)
        key.set_contents_from_filename(f_client)

    def delete(self, f_server):
        key = self.client.get_key(f_server)
        key = key is not None
        self.client.delete_key(f_server)
        return key


class Dropbox_client():
    def __init__(self, credentials=None, bucket=None):
        if credentials is None:
            credentials = op.expanduser('~/.credentials/dropbox.pem')
        self.credentials = credentials
        self.bucket = bucket
        self.client = self.connect()

    def connect(self):
        import dropbox
        if self.credentials is None:
            self.credentials = '~/.credentials/dropbox.pem'
        if isinstance(self.credentials, dict):
            APP_KEY = self.credentials['APP_KEY']
            APP_SECRET = self.credentials['APP_SECRET']
            TOKEN_KEY = self.credentials['TOKEN_KEY']
            TOKEN_SECRET = self.credentials['TOKEN_SECRET']
        elif isinstance(self.credentials, str):
            with open(self.credentials, 'rb') as f:
                auth = f.read()
                auth = auth.split('\n')
                APP_KEY = auth[0].split('APP_KEY=')[1]
                APP_SECRET = auth[1].split('APP_SECRET=')[1]
                TOKEN_KEY = auth[2].split('TOKEN_KEY=')[1]
                TOKEN_SECRET = auth[3].split('TOKEN_SECRET=')[1]
        else:
            raise ValueError('Specify credentials with a file or a dict')
        session = dropbox.session.DropboxSession(
            APP_KEY, APP_SECRET, 'dropbox')
        # request_token = session.obtain_request_token()
        # url = session.build_authorize_url(request_token)
        # access_token = session.obtain_access_token(request_token)
        session.set_token(TOKEN_KEY, TOKEN_SECRET)
        self.client = dropbox.client.DropboxClient(session)

    def metadata(self, f_server):
        from dropbox.client import ErrorResponse
        try:
            metadata = self.client.metadata(
                op.join(self.bucket, f_server))
            metadata['exist'] = True
            if 'is_deleted' in metadata.keys():
                metadata['exist'] = not metadata['is_deleted']
        except ErrorResponse:
            metadata = dict(exist=False, bytes=0)
        return metadata

    def download(self, f_server, f_client):
        f = self.client.get_file(op.join(self.bucket, f_server))
        with open(f_client, 'wb') as out:
            out.write(f.read())

    def upload(self, f_client, f_server):
        f_server = op.join(self.bucket, f_server)
        if self.metadata(f_server)['exist']:
            self.delete(f_server)
        self._upload_chunk(self.client, f_client, f_server)

    def _upload_chunk(self, client, f_client, f_server):
        from StringIO import StringIO
        file_obj = open(f_client, 'rb')
        target_length = os.path.getsize(f_client)
        chunk_size = 10 * 1024 * 1024
        offset = 0
        uploader = client.get_chunked_uploader(file_obj, target_length)
        last_block = None
        params = dict()
        pbar = ProgressBar(target_length, spinner=True)
        error_count = 0
        while offset < target_length:
            if error_count > 3:
                raise RuntimeError
            pbar.update(offset)
            next_chunk_size = min(chunk_size, target_length - offset)
            # read data if last chunk passed
            if last_block is None:
                last_block = file_obj.read(next_chunk_size)
            # set parameters
            if offset > 0:
                params = dict(upload_id=uploader.upload_id, offset=offset)
            try:
                url, ignored_params, headers = client.request(
                    "/chunked_upload", params, method='PUT',
                    content_server=True)
                reply = client.rest_client.PUT(url, StringIO(last_block),
                                               headers)
                new_offset = reply['offset']
                uploader.upload_id = reply['upload_id']
                # avoid reading data if last chunk didn't pass
                if new_offset > offset:
                    offset = new_offset
                    last_block = None
                    error_count == 0
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        if target_length > 0:
            pbar.update(target_length)
        print('')
        file_obj.close()
        uploader.finish(f_server, overwrite=True)

    def delete(self, f_server):
        metadata = self.metadata(f_server)
        if metadata['exist']:
            self.client.file_delete(op.join(self.bucket, f_server))
            return True
        else:
            return False

    def _create_token():
        import dropbox
        credentials = '.credentials/dropbox.pem'
        with open(credentials, 'rb') as f:
            auth = f.read()
            auth = auth.split('\n')
            APP_KEY = auth[0].split('APP_KEY=')[1]
            APP_SECRET = auth[1].split('APP_SECRET=')[1]
        session = dropbox.session.DropboxSession(
            APP_KEY, APP_SECRET, 'dropbox')
        request_token = session.obtain_request_token()
        url = session.build_authorize_url(request_token)
        raw_input(url)
        access_token = session.obtain_access_token(request_token)
        print access_token.key, access_token.secret
        session.set_token(access_token.key, access_token.secret)
        client = dropbox.client.DropboxClient(session)
        print client.account_info()
