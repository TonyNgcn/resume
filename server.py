#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import json
from flask import Flask, request, Response, render_template
from extract.extract import Extractor

extractor = Extractor()

app = Flask(__name__)


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


# HOME页面
@app.route('/', methods=['GET',"POST"])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        resume = request.form.get("resume_input")
        regresume = extractor.single_extract(resume)
        return render_template("home.html", regresume=regresume)


@app.route("/single_extract", methods=["POST"])
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


@app.route("/single_extract_form", methods=["POST"])
def single_extract_form():
    print("test")
    resume = request.form.get("resume")
    regresume = extractor.single_extract(resume)
    return render_template("home.html", regresume=regresume)


@app.route("/batch_extract", methods=["POST"])
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
