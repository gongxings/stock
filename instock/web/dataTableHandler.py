#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


import json
from abc import ABC
from tornado import gen
# import logging
import datetime
import instock.lib.trade_time as trd
import instock.core.singleton_stock_web_module_data as sswmd
import instock.web.base as webBase
from instock.web.baseResponse import BaseResponse

__author__ = 'myh '
__date__ = '2023/3/10 '


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, bytes):
            return "是" if ord(obj) == 1 else "否"
        elif isinstance(obj, datetime.date):
            delta = datetime.datetime.combine(obj, datetime.time.min) - datetime.datetime(1899, 12, 30)
            return f'/OADate({float(delta.days) + (float(delta.seconds) / 86400)})/'  # 86,400 seconds in day
            # return obj.isoformat()
        else:
            return json.JSONEncoder.default(self, obj)


# 获得页面数据。
class GetStockHtmlHandler(webBase.BaseHandler, ABC):
    @gen.coroutine
    def get(self):
        name = self.get_argument("table_name", default=None, strip=False)
        web_module_data = sswmd.stock_web_module_data().get_data(name)
        run_date, run_date_nph = trd.get_trade_date_last()
        if web_module_data.is_realtime:
            date_now_str = run_date_nph.strftime("%Y-%m-%d")
        else:
            date_now_str = run_date.strftime("%Y-%m-%d")
        self.render("stock_web.html", web_module_data=web_module_data, date_now=date_now_str,
                    leftMenu=webBase.GetLeftMenu(self.request.uri))


# 获得股票数据内容。
class GetStockDataHandler(webBase.BaseHandler, ABC):
    def get(self):
        name = self.get_argument("name", default=None, strip=False)
        date = self.get_argument("date", default=None, strip=False)
        web_module_data = sswmd.stock_web_module_data().get_data(name)
        self.set_header('Content-Type', 'application/json;charset=UTF-8')

        if date is None:
            where = ""
        else:
            # where = f" WHERE `date` = '{date}'"
            where = f" WHERE `date` = %s"

        order_by = ""
        if web_module_data.order_by is not None:
            order_by = f" ORDER BY {web_module_data.order_by}"

        order_columns = ""
        if web_module_data.order_columns is not None:
            order_columns = f",{web_module_data.order_columns}"

        sql = f" SELECT *{order_columns} FROM `{web_module_data.table_name}`{where}{order_by}"
        data = self.db.query(sql, date)

        self.write(json.dumps(data, cls=MyEncoder))


class GetStockTableColumnsHandler(webBase.BaseHandler, ABC):
    def get(self):
        name = self.get_argument("name", default=None, strip=False)
        web_module_data = sswmd.stock_web_module_data().get_data(name)
        self.set_header('Content-Type', 'application/json;charset=UTF-8')
        self.write(json.dumps(web_module_data, cls=MyEncoder))


class ListHandler(webBase.BaseHandler, ABC):
    def get(self):
        # 获取查询参数
        page = int(self.get_argument("page", 1))  # 默认第 1 页
        page_size = int(self.get_argument("pageSize", 20))  # 默认每页 20 条

        # 模拟数据库查询
        total_items = 100  # 假设有 100 条数据
        items = [
            {"id": i, "name": f"Item {i}", "description": f"Description of Item {i}"}
            for i in range((page - 1) * page_size, page * page_size)
        ]

        # 构造响应数据
        response_data = {
            "items": items,
            "total": total_items
        }

        # 返回 JSON 响应
        response = BaseResponse(
            code=0,
            data=response_data,
            message="ok"
        )
        self.write(response.to_json())
