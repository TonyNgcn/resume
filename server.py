#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import json
from flask import Flask, Blueprint, request, Response
from extract.extract import Extractor

extractor = Extractor()

app = Flask(__name__)

# api_bp = Blueprint("api", __name__)
#
# app.register_blueprint(api_bp, url_prefix="/api")


def success(data):
    body = {
        "code": 1,
        "msg": "操作成功",
        "data": data,
    }
    return json.dumps(body, ensure_ascii=False)


def fail(msg: str):
    body = {
        "code": 0,
        "msg": msg,
    }
    return json.dumps(body, ensure_ascii=False)


@app.route("/single_extract", methods=["post"])
def single_extract():
    resume = request.json["resume"]
    if isinstance(resume, str):
        if resume != "":
            regresume = extractor.single_extract(resume)
            data = {
                "regresume": regresume,
            }
            return Response(success(data), mimetype="application/json")
        else:
            return Response(fail("resume 不能为空"), mimetype="application/json")
    else:
        return Response(fail("resume 格式错误"), mimetype="application/json")


@app.route("/batch_extract", methods=["post"])
def batch_extract():
    resumes = request.json["resumes"]
    if isinstance(resumes, list):
        if len(resumes) > 0:
            regresumes = extractor.batch_extract(resumes)
            data = {
                "regresumes": regresumes,
            }
            return Response(success(data), mimetype="application/json")
        else:
            return Response(fail("resumes 不能为空"), mimetype="application/json")
    else:
        return Response(fail("resumes 格式错误"), mimetype="application/json")


if __name__ == '__main__':
    logging.info("开启服务器")
    app.run(host="127.0.0.1", port=11000, debug=False)
