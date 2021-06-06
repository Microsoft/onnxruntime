// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// import {NODE_TESTS_ROOT, warmup} from './test-utils';

// // warmup
// //
// // for unknown reason, the first call to native InferenceSession::Run() is very slow.
// // we need this warmup call so that coming test cases will not fail because of timeout.
// warmup();

// // unittests
// require('./unittests/lib/index');
// require('./unittests/lib/inference-session');
// require('./unittests/lib/tensor');

// // E2E tests
// require('./e2e/simple-e2e-tests');
// require('./e2e/inference-session-run');

// // Test ONNX spec tests
// import {run as runTestRunner} from './test-runner';
// describe('ONNX spec tests', () => {
//   runTestRunner(NODE_TESTS_ROOT);
// });
