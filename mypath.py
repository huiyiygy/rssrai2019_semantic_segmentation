# -*- coding:utf-8 -*-
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == "rssrai2019":
            return '/home/lab/ygy/rssrai2019/datasets'  # folder that contains rssrai2019
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
