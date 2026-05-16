param(
    [string]$BackendBaseUrl = 'http://127.0.0.1:8000',
    [string]$FrontendProxyUrl = 'http://127.0.0.1:5173',
    [int]$ChatThrottleSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$B = $BackendBaseUrl.TrimEnd('/')
$Fx = $FrontendProxyUrl.TrimEnd('/')
$MdPath = Join-Path $RepoRoot 'SUPERVISOR_DETAILED_API_REPORT.md'

$Bf = Join-Path $env:TEMP ('sup_' + ([guid]::NewGuid().ToString()) + '.txt')
$UtfNb = New-Object System.Text.UTF8Encoding $false

$Sb = [System.Text.StringBuilder]::new()
function L([string]$s) { [void]$Sb.AppendLine($s) }

function FBlock([string]$title, [string]$langLower, [string]$content) {
    L ''
    L $title
    L ''
    L ('```' + $langLower)
    if (($null -ne $content) -and ($content.Length -gt 0)) { L $content.TrimEnd("`r") }
    L '```'
}

function CurlRun([string[]]$TailAfterOut) {
    if (Test-Path $Bf) { Remove-Item $Bf -Force -ErrorAction Stop }
    $common = @(
        '-s', '--max-time', '420', '--connect-timeout', '30',
        '-w', '%{http_code}', '-o', $Bf
    )
    $all = $common + $TailAfterOut
    $sx = & curl.exe @all 2>$null | Out-String
    $code = -1
    try { $code = [int]$sx.Trim() } catch { $code = -1 }
    $body = ''
    if (Test-Path $Bf -PathType Leaf) {
        $body = [System.IO.File]::ReadAllText($Bf, $UtfNb)
    }
    return [PSCustomObject]@{ Status = $code; BodyText = $body }
}

try {
    L '# Детализированный протокол проверки HTTP API assistant_rag'
    L ''
    L '**Цель документа.** Материал для научного руководителя: воспроизводимые примеры HTTP-запросов к приложению, полное тело обращений и без сокращений полное тело ответов сервера вместе с кодами статуса.'
    L ''
    L ("**Отметка времени (локальная ОС, момент запуска генератора протокола).** {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date))
    L ("**Backend (FastAPI, Docker Compose).** {0}" -f $B)
    L ("**Frontend (nginx, смок ниже через прокси /api/).** {0}" -f $Fx)
    L ''
    L ("Ограничение на использование платного/free API Gemini между сценариями чата: после завершения **каждого** `POST /chat` следующий `POST /chat` отправляется **не ранее чем через {0} секунд**, кроме ситуации «нет предыдущего `POST /chat` в этом прогоне» — перед первым чат-сценарием секунду ожидания не ждём." -f $ChatThrottleSeconds)
    L ''
    L 'Для эндпоинтов **`GET`** и последовательных операций загрузки файлов **`POST /upload`**, а также для финального запроса к фронту, дополнительной минутной задержке между ними мы **не следуем**, поскольку они не включают прямой синтез ответов модели Gemini из цепочки чата так же, как `POST /chat`.'
    L ''
    L '---'

    # 1
    L ''
    L '## Запись 1. GET /health'
    L ''
    L ('* Абсолютный URL (воспроизведение): `' + $B + '/health`')
    L ''
    L '* Пример утилиты curl для повторной проверки: то же, что выполняло скрипт — GET; ключ `-w "%{http_code}"` отправляет код состояния в stdout, а тело в файл оператора через `-o` (символ переназначения см. официально справке curl под Windows):'
    L ''
    $r1 = CurlRun @(($B + '/health'))
    L '* Наблюдаемый код ответа HTTP:'
    L (('`' + '{0}' + '`') -f $r1.Status)
    FBlock '* Полный текст ответа (поле тела сообщения без обрезки)' 'txt' $r1.BodyText

    # 2
    L ''
    L '## Запись 2. GET /ready'
    L ''
    L ('* Абсолютный URL: `' + $B + '/ready`')
    L ''
    $r2 = CurlRun @(($B + '/ready'))
    L '* Наблюдаемый код ответа HTTP:'
    L (('`' + '{0}' + '`') -f $r2.Status)
    FBlock '* Полный текст ответа (JSON-документ о готовности Qdrant/Ollama согласно коду приложения)' 'json' $r2.BodyText

    # uploads
    $pairs = @(
        @{ Idx = '3'; File = Join-Path $RepoRoot 'data\docker_test_doc.txt'; Label = '`docker_test_doc.txt`' },
        @{ Idx = '4'; File = Join-Path $RepoRoot 'data\e2e_upload_verify.txt'; Label = '`e2e_upload_verify.txt`' }
    )
    foreach ($u in $pairs) {
        L ''
        L ('## Запись ' + $u.Idx + '. POST /upload для файла ' + $u.Label)
        L '* HTTP method: POST, Content-Type multipart/form-data, полевое имя `file`.'
        FBlock '* Абсолютный путь к переданному бинарю на машине генерации отчета' 'txt' $u.File
        $r = CurlRun @('-X', 'POST', ($B + '/upload'), '-F', ('file=@' + $u.File))
        L '* Наблюдаемый код ответа HTTP:'
        L (('`' + '{0}' + '`') -f $r.Status)
        FBlock '* Полный текст тела ответа' 'txt' $r.BodyText
    }

    # invalid
    L ''
    L '## Запись 5. POST /upload для файла недопустимого расширения'
    $bad = Join-Path $RepoRoot 'data\raw\_supervisor_report_bad.invalid'
    'x' | Out-File $bad -Encoding ascii -NoNewline
    FBlock '* Абсолютный путь к искусственно созданному файлу ошибки' 'txt' $bad
    $badR = CurlRun @('-X', 'POST', ($B + '/upload'), '-F', ('file=@' + $bad))
    Remove-Item $bad -Force -ErrorAction SilentlyContinue
    L '* Наблюдаемый код ответа HTTP (ожидание 400):'
    L (('`' + '{0}' + '`') -f $badR.Status)
    FBlock '* Полное тело ответа ошибки' 'json' $badR.BodyText

    # chats 6..
    $chats = @(
        @{ T = 'короткое приветствие (routing direct)'        ; Json = Join-Path $RepoRoot 'data\e2e_chat\direct.json' }
        @{ T = 'арифметический калькулятор'                        ; Json = Join-Path $RepoRoot 'data\e2e_chat\calc.json' }
        @{ T = 'погода (OpenWeatherMap через backend-тулинг)' ; Json = Join-Path $RepoRoot 'data\e2e_chat\weather.json' }
        @{ T = 'энциклопедия (ветка Wikipedia)'              ; Json = Join-Path $RepoRoot 'data\e2e_chat\wiki.json' }
        @{ T = 'подбор образовательного контента по YouTube' ; Json = Join-Path $RepoRoot 'data\e2e_chat\youtube.json' }
        @{ T = 'актуальный веб-поиск'                          ; Json = Join-Path $RepoRoot 'data\e2e_chat\web.json' }
        @{ T = 'RAG-поиск по загруженным пользователем файлам'; Json = Join-Path $RepoRoot 'data\e2e_chat\rag.json' }
    )

    $rec = 6
    for ($i = 0; $i -lt $chats.Count; $i++) {
        $it = $chats[$i]
        if (-not (Test-Path $it.Json)) { throw ('Fixture absent: ' + $it.Json) }

        if ($i -gt 0) {
            L ''
            L '---'
            L (('>Пауза **{0} с** между **предыдущим завершением** запросов `POST /chat` и **следующим** («{1}»).' -f [int]$ChatThrottleSeconds, $it.T))
            Write-Host ('WAIT ' + $ChatThrottleSeconds + ' seconds before POST /chat: ' + $it.T)
            Start-Sleep -Seconds $ChatThrottleSeconds
        }

        $reqTxt = [System.IO.File]::ReadAllText($it.Json, $UtfNb)
        L ''
        L ('## Запись ' + $rec + '. POST /chat — `' + $it.T + '`')
        L '* Абсолютный адрес:'
        L (('`' + $B + '/chat`'))
        L '* Заголовок Content-Type:'
        L '`application/json; charset=UTF-8`'
        L '* Сервер приложения ожидает объект с полем `text` пользовательского сообщения (смотрите файл фиксации JSON в репозитории того же содержания). Ниже вставлена копия тела отправки:'
        FBlock '* Полное тело JSON-запроса отправленное curl' 'json' $reqTxt

        $rc = CurlRun @(
            '-X', 'POST', ($B + '/chat'), '-H', 'Content-Type: application/json;charset=UTF-8',
            '--data-binary', ('@' + $it.Json)
        )
        L '* Наблюдаемый код ответа HTTP:'
        L (('`' + '{0}' + '`') -f $rc.Status)
        FBlock '* Полный JSON-документ ответа приложения клиенту' 'json' $rc.BodyText
        $rec++
    }

    # frontend smoke
    L ''
    L (('## Запись ' + $rec + '. GET /api/health через nginx фронта (прокси на backend `/health`)'))
    L ''
    L ('* Абсолютный URL: `' + $Fx + '/api/health`')
    $rf = CurlRun @( ($Fx + '/api/health') )
    L (('Наблюдаемый HTTP status: **`{0}`**' ) -replace '\{0\}', $rf.Status)
    FBlock '* Полный текст тела смок-проверки' 'txt' $rf.BodyText

    L ''
    L '---'
    L ''
    L '## Приложение. Примечание к контролируемому результату последнего RAG-сценария из реестра данных'
    L ''
    L 'При корректной работе связки загрузчик → Qdrant → агент, ответ последнего `POST /chat` на тест из `rag.json` в поле строки пользователя обычно содержит уникальный токен текстового содержания из соответствующего файлового артефакта `data/e2e_upload_verify.txt` (подробное значение смотреть исключительно в полном JSON-ответе выше записи этого сценария).'
}
finally {
    if (Test-Path $Bf) { Remove-Item $Bf -Force -ErrorAction SilentlyContinue }
}

[System.IO.File]::WriteAllText($MdPath, $Sb.ToString(), $UtfNb)
Write-Host ('Report saved: ' + $MdPath)
