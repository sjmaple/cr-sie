local json = require("cjson.safe")
json.decode_invalid_numbers(false)
json.decode_max_depth(32)

local allowed = {
    ["Qwen/Qwen3.5-4B"] = true,
    ["Qwen/Qwen3.8-27B-FP8"] = true,
}

local function unique_top_level_keys(body)
    local seen, depth, previous, i = {}, 0, "", 1
    while i <= #body do
        local c = body:sub(i, i)
        if c == '"' then
            local start = i
            i = i + 1
            while i <= #body do
                local q = body:sub(i, i)
                if q == "\\" then
                    i = i + 2
                elseif q == '"' then
                    break
                else
                    i = i + 1
                end
            end
            if depth == 1 and (previous == "{" or previous == ",") then
                local key = json.decode(body:sub(start, i))
                if type(key) ~= "string" or seen[key] then
                    return false
                end
                seen[key] = true
            end
            previous = '"'
        elseif c == "{" or c == "[" then
            depth = depth + 1
            previous = c
        elseif c == "}" or c == "]" then
            depth = depth - 1
            previous = c
        elseif not c:match("%s") then
            previous = c
        end
        i = i + 1
    end
    return true
end

core.register_action("validate-body", {"http-req"}, function(txn, kind)
    txn:set_var("txn.valid_body", false)
    local body = txn.sf:req_body()
    local length = txn.sf:req_hdr("content-length")
    if type(body) ~= "string" or not length:match("^%d+$")
        or tonumber(length) ~= #body or #body == 0 or #body > 131072 then
        return
    end
    if kind == "chat" then
        local value = json.decode(body)
        if body:match("^%s*(.)") ~= "{" or type(value) ~= "table"
            or type(value.model) ~= "string" or not allowed[value.model]
            or not unique_top_level_keys(body) then
            return
        end
        txn:set_var("txn.model", value.model)
    end
    txn:set_var("txn.valid_body", true)
end, 1)
