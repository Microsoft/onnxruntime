// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "environment.h"
#include "http_server.h"
#include "json_handling.h"
#include "executor.h"

namespace onnxruntime {
namespace hosting {

void BadRequest(HttpContext& context, const std::string& error_message) {
  auto json_error = R"({"error_code": 400, "error_message": )" + error_message + " }";

  context.response.result(400);
  context.response.body() = std::string(json_error);
}

// TODO: decide whether this should be a class
void Predict(const std::string& name,
             const std::string& version,
             const std::string& action,
             HttpContext& context,
             std::shared_ptr<HostingEnvironment> env) {
  PredictRequest predictRequest{};
  auto logger = env->GetLogger();

  LOGS(logger, VERBOSE) << "Name: " << name;
  LOGS(logger, VERBOSE) << "Version: " << version;
  LOGS(logger, VERBOSE) << "Action: " << action;

  auto body = context.request.body();
  auto status = GetRequestFromJson(body, predictRequest);

  if (!status.ok()) {
    return BadRequest(context, status.error_message());
  }

  Executor executor(env);
  PredictResponse response{};
  executor.predict(name, version, "request_id", predictRequest, response);

  std::string response_body{};
  status = GenerateResponseInJson(response, response_body);
  context.response.body() = response_body;
  context.response.result(200);
  context.response.set(http::field::content_type, "application/json");
};

}  // namespace hosting
}  // namespace onnxruntime
