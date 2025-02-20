;;; gptel-bedrock.el --- AWS Bedrock support for gptel  -*- lexical-binding: t; -*-

;; Copyright (C) 2024  [Your Name]

;; Author: [Your Name] <[your-email@example.com]>
;; Keywords: comm, convenience

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:

;; This file adds support for AWS Bedrock to gptel.  Documentation for the request data and the
;; response payloads can be found at these two links:
;; * https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
;; * https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html

;;; Code:
(require 'cl-generic)
(require 'map)
(require 'gptel)

(defvar json-object-type)

(cl-defstruct (gptel-bedrock (:constructor gptel--make-bedrock)
                             (:copier nil)
                             (:include gptel-backend)))

(defconst gptel-bedrock--prompt-type
  ;; For documentation purposes only -- this describes the type of prompt objects that get passed
  ;; around. https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html
  '(plist
    :role (member "user" "assistant")
    :content (array (or (plist :text string)
                        (plist :toolUse (plist :input any :name string :toolUseId string))
                        (plist :toolResult (plist
                                            :toolUseId string
                                            :status (member "success" "error")
                                            ;; AWS allows more result types in
                                            ;; ToolResultContentBlock, but we only send text results
                                            :content (array (plist :text string))))
                        ))))

(cl-defmethod gptel--request-data ((backend gptel-bedrock) prompts)
  "Prepare request data for AWS Bedrock in converse format from PROMPTS."
  (nconc
   `(:messages [,@prompts] :inferenceConfig (:maxTokens ,(or gptel-max-tokens 500)))
   (when gptel--system-message `(:system ,gptel--system-message))
   (when gptel-temperature `(:temperature ,gptel-temperature))
   (when (and gptel-use-tools gptel-tools)
     `(:toolConfig (:toolChoice ,(if (eq gptel-use-tools 'force) "any" "auto")
                    :tools ,(gptel--parse-tools backend gptel-tools))))))

(cl-defmethod gptel--parse-tools ((_backend gptel-bedrock) tools)
  "Parse TOOLS and return a list of ToolSpecification objects.

TOOLS is a list of `gptel-tool' structs, which see."
  ;; https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolSpecification.html
  (vconcat
   (mapcar
    (lambda (tool)
      (list
       :name (gptel-tool-name tool)
       :description (gptel-tool-description tool)
       :inputSchema (gptel--tool-args-to-json-schema (gptel-tool-args tool))))
    (ensure-list tools))))

(cl-defmethod gptel--parse-response ((_backend gptel-bedrock) response info)
  "Parse a Bedrock (non-streaming) RESPONSE and return response text.

Mutate state INFO with response metadata."
  (plist-put info :stop-reason (plist-get response :stopReason))
  (plist-put info :input-tokens
             (map-nested-elt response '(:usage :inputTokens)))
  (plist-put info :output-tokens
             (map-nested-elt response '(:usage :outputTokens)))

  (let* ((message (map-nested-elt response '(:output :message)))
         (content (plist-get message :content))
         (content-strs (thread-last content
                                    (mapcar (lambda (cblock) (plist-get cblock :text)))
                                    (delq nil)))
         (tool-use (cl-remove-if-not
                    (lambda (cblock) (plist-get cblock :toolUse))
                    content)))
    (when tool-use
      (cl-callf (lambda (prompts) (vconcat prompts (list message)))
          (plist-get (plist-get  info :data) :messages))

      (plist-put info :tool-use (mapcar (lambda (call)
                                          (list
                                           :name (plist-get call :name)
                                           :args (plist-get call :input)
                                           :id (plist-get call :toolUseId)))
                                        tool-use)))
    (and content-strs (apply #'concat content-strs))))

(cl-defmethod gptel--parse-list ((_backend gptel-anthropic) prompt-strings)
  "Create a list of prompt objects from PROMPT-STRINGS.

Assumes this is a conversation with alternating roles."
  (cl-loop for text in prompt-strings
           for role = t then (not role)
           if text collect
           (list :role (if role "user" "assistant")
                 :content `[(:text ,text)])))

(cl-defmethod gptel--wrap-user-prompt ((_backend gptel-openai) prompts &optional inject-media)
  "Wrap the last user prompt in PROMPTS with the context string.

If INJECT-MEDIA is non-nil wrap it with base64-encoded media
files in the context."
  (if inject-media
      (error "Not implemented yet") ; TODO

    (cl-callf (lambda (current)
                (cl-etypecase current
                  (string (gptel-context--wrap current))
                  (vector (if-let* ((wrapped (gptel-context--wrap nil)))
                              (vconcat `[(:text ,wrapped)] current)
                            current))))
        (plist-get (car (last prompts)) :content))))

(defvar-local gptel-bedrock--stream-cursor nil
  "Marker to indicate last point parsed.")

(cl-defmethod gptel-curl--parse-stream ((_backend gptel-bedrock) info)
  "Parse an AWS Bedrock streaming response from the ConverseStream API.
INFO is a plist containing the request context."
  (save-excursion
    ;; TODO: Does gptel-curl recycle buffers? Need to figure out how to clear out the cursor if so
    (when (null gptel-bedrock--stream-cursor)
      (goto-char (point-min))
      ;; Only initialize the cursor once we have headers
      (if (search-forward "\r\n\r\n" nil t)
          (progn
            ;; TODO: Add assertion that content-type is application/vnd.amazon.eventstream
            (setq gptel-bedrock--stream-cursor (make-marker))
            (move-marker gptel-bedrock--stream-cursor (point)))
        (cl-return)))

    (let ((inhibit-modification-hooks t)
          (buffer-undo-list t))
      ;; TODO: Where is the right place for this? Need to ensure we're in unibyte mode for binary
      ;; data handling
      (set-buffer-multibyte nil))

    (let ((content-strs))
      (goto-char gptel-bedrock--stream-cursor)
      (while-let ((message (gptel-bedrock--parse-stream-message)))
        (push (gptel-bedrock--handle-stream-message message info) content-strs))
      (move-marker gptel-bedrock--stream-cursor (point))

      (apply #'concat (nreverse content-strs)))))

(defun gptel-bedrock--parse-stream-message ()
  "Parse AWS Bedrock event-stream message starting at current position.
Point should be at the beginning of an event in the `vnd.amazon.event-stream'
format.  Returns plist with :headers and :payload keys if successful, nil if
incomplete."
  ;; https://github.com/awslabs/aws-c-event-stream has documentation of this format
  ;; The format consists of three main sections: Prelude, Data, and Message CRC.
  ;; 1. Prelude (12 bytes)
  ;;    a. Total Byte Length (4 bytes): Specifies the total length of the message.
  ;;    b. Headers Byte Length (4 bytes): Indicates the length of the headers section.
  ;;    c. Prelude CRC (4 bytes): A CRC value for validating the integrity of the prelude.
  ;; 2. Data (variable length)
  ;;    a. Headers: An array of packed headers. Each header has a specific format documented in
  ;;       gptel-bedrock--parse-headers
  ;;    b. Payload: The main message content, also of variable length. Length can be computed from
  ;;       the prelude fields by subtracting the prelude length, headers length, and message CRC
  ;;       length from the total.
  ;; 3. Message CRC (4 bytes): A 4-byte CRC to validate the integrity of the entire message.

  ;; (point-max) is the position after the last character, hence the use of >= and not > below
  (when (>= (- (point-max) (point)) 12)
    (let* ((prelude-start (point))
           (prelude-length 12)
           (prelude-end (+ prelude-start prelude-length))
           (prelude (buffer-substring-no-properties prelude-start prelude-end))
           (total-length (gptel-bedrock--bytes-to-int32 (substring prelude 0 4)))
           (headers-length (gptel-bedrock--bytes-to-int32 (substring prelude 4 8)))
           ;; TODO validate prelude CRC?
           (headers-start prelude-end)
           (headers-end (+ headers-start headers-length))
           headers payload)

      (when (>= (point-max) (+ prelude-start total-length))
        (goto-char headers-start)
        (setq headers (gptel-bedrock--parse-headers (buffer-substring (point) headers-end)))
        (cl-assert (equal (assoc-default ":content-type" headers) "application/json")
                   t "Unexpected content-type %S is not %S")

        (goto-char headers-end)
        (setq payload (gptel--json-read))
        (let* ((message-crc 4)
               (payload-length (- total-length headers-length prelude-length message-crc)))
          (cl-assert (= (- (point) headers-end) payload-length)
                     t "Unexpected payload length %d; expected %d."))

        ;; TODO validate message CRC
        (goto-char (+ prelude-start total-length))
        `(:headers ,headers :payload ,payload)))))

(defun gptel-bedrock--parse-headers (headers-data)
  "Parse HEADERS-DATA into alist of (NAME . VALUE).
Keys are string-valued, lower-cased names."
  ;; Header wire format:
  ;;   1. Header Name Byte Length (1 byte): Specifies the length of the header name.
  ;;   2. Header Name (String) (Variable length): Contains the name of the header.
  ;;   3. Header Value Type (1 byte): Identifies the type of the header value.
  ;;   4. Value String Byte Length (2 bytes): Indicates the length of the value string.
  ;;   5. Value (Variable length): Holds the actual value bytes
  (let ((pos 0) (max (length headers-data)) headers)
    (cl-flet ((pos++ (&optional n) (prog1 pos (setq pos (+ pos (or n 1)))))
              (++pos (&optional n) (cl-incf pos n))
              (utf8 (unibyte-string) (decode-coding-string unibyte-string 'utf-8 t)))
      (while (< pos max)
        (let* ((name-len (aref headers-data (pos++)))
               (name (substring headers-data pos (++pos name-len)))
               (type (aref headers-data (pos++)))
               (value-len (gptel-bedrock--bytes-to-int16 (substring headers-data pos (++pos 2))))
               (value
                (pcase type
                  ;; Header types from https://awslabs.github.io/aws-crt-python/api/eventstream.html
                  (0 t)
                  (1 :json-false)
                  (2 (let ((res (aref headers-data (pos++)))) ; int8
                       (if (> res 127) (- res 256) res)))
                  (3 (gptel-bedrock--bytes-to-int16 (substring headers-data pos (++pos 2)))) ; int16
                  (4 (gptel-bedrock--bytes-to-int32 (substring headers-data pos (++pos 4)))) ; int32
                  (5 (gptel-bedrock--bytes-to-int64 (substring headers-data pos (++pos 8)))) ; int64
                  (6 (substring headers-data pos (++pos value-len)))                         ; raw bytes
                  (7 (utf8 (substring headers-data pos (++pos value-len))))                  ; utf8 string
                  (8 (decode-time       ; 64 bit int with seconds since the Unix epoch
                      (gptel-bedrock--bytes-to-int64 (substring headers-data pos (++pos 8))) t))
                  (9 (gptel-bedrock--bytes-to-uuid ; 16 byte UUID
                      (substring headers-data pos (++pos 16))))
                  (_ (error "Unknown header type: %d" type)))))
          (push (cons (downcase (utf8 name)) value) headers)))
      (cl-assert (= pos max) t "Headers did not parse cleanly. pos=%d  header-len=%d")
      headers)))

(defun gptel-bedrock--bytes-to-int16 (bytes)
  "Convert 2-byte string BYTES to big-endian signed integer."
  (let ((b0 (logand (aref bytes 0) 255))
        (b1 (logand (aref bytes 1) 255)))
    (let ((result (+ (ash b0 8) b1)))
      (if (>= b0 #x80) (- result (ash 1 16)) result))))

(defun gptel-bedrock--bytes-to-int32 (bytes)
  "Convert 4-byte string BYTES to big-endian signed integer."
  (let ((b0 (logand (aref bytes 0) 255))
        (b1 (logand (aref bytes 1) 255))
        (b2 (logand (aref bytes 2) 255))
        (b3 (logand (aref bytes 3) 255)))
    (let ((result (+ (ash b0 24) (ash b1 16) (ash b2 8) b3)))
      (if (>= b0 #x80) (- result (ash 1 32)) result))))

(defun gptel-bedrock--bytes-to-int64 (bytes)
  "Convert 8-byte string BYTES to big-endian signed integer."
  (let ((b0 (logand (aref bytes 0) 255))
        (b1 (logand (aref bytes 1) 255))
        (b2 (logand (aref bytes 2) 255))
        (b3 (logand (aref bytes 3) 255))
        (b4 (logand (aref bytes 4) 255))
        (b5 (logand (aref bytes 5) 255))
        (b6 (logand (aref bytes 6) 255))
        (b7 (logand (aref bytes 7) 255)))
    (let ((result-u63 (+ (ash (logand b0 #x7f) 56) (ash b1 48)
                         (ash b2 40) (ash b3 32) (ash b4 24) (ash b5 16) (ash b6 8) b7)))
      (if (>= b0 #x80)
          (- result-u63 (ash 1 63))
        result-u63))))

(defun gptel-bedrock--bytes-to-uuid (bytes)
  "Convert a 16-byte unibyte BYTES to a 36 character UUID string."
  (unless (and (stringp bytes) (= (length bytes) 16))
    (error "Input must be a 16-byte unibyte string"))
  (let ((hex (mapconcat (lambda (i) (format "%02x" (aref bytes i))) (number-sequence 0 15) "")))
    (format "%s-%s-%s-%s-%s"
            (substring hex 0 8)
            (substring hex 8 12)
            (substring hex 12 16)
            (substring hex 16 20)
            (substring hex 20 32))))

(defun gptel-bedrock--handle-stream-message (message info)
  "Process a complete MESSAGE from a response stream and return the text responses.
Mutates INFO to capture tool use info and other metadata."
  (let* ((headers (plist-get message :headers))
         (message-type (assoc-default ":message-type" headers))
         (event-type (assoc-default ":event-type" headers))
         (payload (plist-get message :payload))
         (tool-use (plist-get info :tool-use))
         content-strs)
    (cl-assert (equal message-type "event") nil "Unknown message type %S" message-type)

    (pcase event-type
      ("metadata"
       (plist-put info :input-tokens (map-nested-elt payload '(:usage :inputTokens)))
       (plist-put info :output-tokens (map-nested-elt payload '(:usage :outputTokens))))

      ("messageStart" ;; TODO: Do we need to extract :role?
       )

      ("contentBlockDelta"
       (let ((delta (map-nested-elt payload '(:delta :text))))
         (when delta (push delta content-strs))

         ;; TODO: Fix this tool-call handler
         (when-let ((tool-call (map-nested-elt payload '(:delta :toolUse))))
           (let* ((tool-id (plist-get tool-call :toolUseId))
                  (existing (alist-get tool-id tool-use nil nil #'equal)))
             (if existing
                 (setf (alist-get tool-id tool-use nil nil #'equal)
                       (plist-put existing :input
                                  (concat (plist-get existing :input)
                                          (plist-get tool-call :input))))
               (push (cons tool-id
                           (list :name (plist-get tool-call :name)
                                 :input (plist-get tool-call :input)))
                     tool-use))))))

      ("messageStop"
       (plist-put info :stop-reason (plist-get payload :stopReason))))

    (apply #'concat (nreverse content-strs))))

(cl-defmethod gptel--parse-buffer ((_backend gptel-bedrock) &optional max-entries)
  "Parse current buffer and return a list of prompts for Bedrock.

MAX-ENTRIES  is the maximum number of prompts to include."
  (when gptel-track-media (error "Media tracking is not implemented yet")) ; TODO
  (unless max-entries (setq max-entries most-positive-fixnum))
  (let ((prompts nil) (prev-pt (point)))
    (cl-flet ((capture-prompt (role beg end)
                (let* ((text (gptel--trim-prefixes (buffer-substring-no-properties beg end)))
                       (prompt (list :role role :content `[(:text ,text)])))
                  (push prompt prompts))))

      (if (or gptel-mode gptel-track-response)
          (while (and (> max-entries 0)
                      (/= prev-pt (point-min))
                      (goto-char (previous-single-property-change
                                  (point) 'gptel nil (point-min))))
            (capture-prompt (pcase (get-char-property (point) 'gptel)
                              ('response "assistant")
                              ('nil "user"))
                            (point) prev-pt)
            (setq prev-pt (point))
            (cl-decf max-entries))
        (capture-prompt "user" (point-min) (point-max)))
      prompts)))

;; gptel--inject-prompt not needed since the default implementation works here

(cl-defmethod gptel--parse-tool-results ((_backend gptel-bedrock) tool-use-requests)
  "Return a BACKEND-appropriate prompt containing tool call RESULTS.

This will be injected into the messages list in the prompt to send to the LLM."
  `(:role "user"
    :content
    (vconcat
     ,(mapcar
       (lambda (tool-call)
         `(:toolResult (:toolUseId ,(plist-get tool-call :id)
                        :status ,(if (plist-get tool-call :tool-success) "success" "error")
                        :content [(:text ,(plist-get tool-call :result))])))
       tool-use-requests))))

(defconst gptel--bedrock-models
  ;; TODO: fill this out
  '()
  "List of available AWS Bedrock models and associated properties.")

(defun gptel-bedrock--get-credentials ()
  "Return the AWS credentials to use for the request.

Returns a list of 2-3 elements, depending on whether a session
token is needed, with this form: (AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN).

Convenient to use with `cl-multiple-value-bind'"
  (let ((key-id (getenv "AWS_ACCESS_KEY_ID"))
        (secret-key (getenv "AWS_SECRET_ACCESS_KEY"))
        (token (getenv "AWS_SESSION_TOKEN")))
    (cond
     ((and key-id secret-key token) (cl-values key-id secret-key token))
     ((and key-id secret-key) (cl-values key-id secret-key))
     ;; TODO: Add support for more credential sources
     (t (user-error "Missing AWS credentials; currently only environment variables are supported")))))

(defvar gptel-bedrock-model-ids
  ;; https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
  '((claude-3-5-sonnet-20241022  . "anthropic.claude-3-5-sonnet-20241022-v2:0")
    (claude-3-5-sonnet-20240620  . "anthropic.claude-3-5-sonnet-20240620-v1:0")
    (claude-3-5-haiku-20241022   . "anthropic.claude-3-5-haiku-20241022-v1:0")
    (claude-3-opus-20240229      . "anthropic.claude-3-opus-20240229-v1:0")
    (claude-3-sonnet-20240229    . "anthropic.claude-3-sonnet-20240229-v1:0")
    (claude-3-haiku-20240307     . "anthropic.claude-3-haiku-20240307-v1:0")
    (mistral-7b                  . "mistral.mistral-7b-instruct-v0:2")
    (mistral-8x7b                . "mistral.mixtral-8x7b-instruct-v0:1")
    (mistral-large-2402          . "mistral.mistral-large-2402-v1:0")
    (mistral-large-2407          . "mistral.mistral-large-2407-v1:0")
    (mistral-small-2402          . "mistral.mistral-small-2402-v1:0")
    (llama-3-8b                  . "meta.llama3-8b-instruct-v1:0")
    (llama-3-70b                 . "meta.llama3-70b-instruct-v1:0")
    (llama-3-1-8b                . "meta.llama3-1-8b-instruct-v1:0")
    (llama-3-1-70b               . "meta.llama3-1-70b-instruct-v1:0")
    (llama-3-1-405b              . "meta.llama3-1-405b-instruct-v1:0")
    (llama-3-2-1b                . "meta.llama3-2-1b-instruct-v1:0")
    (llama-3-2-3b                . "meta.llama3-2-3b-instruct-v1:0")
    (llama-3-2-11b               . "meta.llama3-2-11b-instruct-v1:0")
    (llama-3-2-90b               . "meta.llama3-2-90b-instruct-v1:0")
    (llama-3-3-70b               . "meta.llama3-3-70b-instruct-v1:0"))
  "Map of model name to bedrock id.

IDs can be added or replaced by calling
\(push (model-name . \"model-id\") gptel-bedrock-model-ids).")

(defun gptel-bedrock--get-model-id (model)
  "Return the Bedrock model ID for MODEL."
  (or (alist-get model gptel-bedrock-model-ids nil nil #'eq)
      (error "Unknown Bedrock model: %s" model)))

;;;###autoload
(cl-defun gptel-make-bedrock
    (name &key
          region
          (models gptel--bedrock-models)
          (stream nil)
          (protocol "https"))
  "Register an AWS Bedrock backend for gptel with NAME.

Keyword arguments:

REGION - AWS region name (e.g. \"us-east-1\")
MODELS - The list of models supported by this backend
STREAM - Whether to use streaming responses or not."
  (declare (indent 1))
  (let ((host (format "bedrock-runtime.%s.amazonaws.com" region)))
    (setf (alist-get name gptel--known-backends nil nil #'equal)
          (gptel--make-bedrock
           :name name
           :host host
           :header nil           ; x-amz-security-token is set in curl-args if needed
           :models (gptel--process-models models)
           :protocol protocol
           :endpoint "" ; Url is dynamically constructed based on other args
           :stream stream
           :curl-args
           (lambda ()
             ;; https://how.wtf/aws-sigv4-requests-with-curl.html
             (cl-multiple-value-bind (key-id secret token) (gptel-bedrock--get-credentials)
               (concat
                `("--user" ,(format "%s:%s" key-id secret)
                  "--aws-sigv4" ,(format "aws:amz:%s:bedrock" region))
                (when token
                  (list (format "-Hx-amz-security-token: %s" token))))))
           :url
           (lambda ()
             (concat protocol "://" host
                     "/model/" (gptel-bedrock--get-model-id gptel-model)
                     "/" (if stream "converse-stream" "converse")))
           ))))

(provide 'gptel-bedrock)
;;; gptel-bedrock.el ends here
