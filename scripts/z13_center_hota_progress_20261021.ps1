param([string]$Run='E:\Dissertation\runs\z13_center_hota_20261021')
function Read-Json([string]$Path) {if(Test-Path -LiteralPath $Path){try{return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json -DateKind String}catch{return $null}}}
$Host.UI.RawUI.WindowTitle='FlyVis — центрированный считыватель в полном трекере'
@{pid=$PID;started=[DateTimeOffset]::Now.ToString('o');run=$Run} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'progress_console.json') -Encoding utf8
while($true){
    Clear-Host
    $p=Read-Json (Join-Path $Run 'protocol.json');$s=Read-Json (Join-Path $Run 'status.json');$q=Read-Json (Join-Path $Run 'queue_status.json');$active=Read-Json (Join-Path $Run 'active_job.json')
    Write-Host 'FlyVis: проверка центрированного считывателя в трекере' -ForegroundColor Cyan
    Write-Host ('Время МСК: '+[DateTimeOffset]::Now.ToString('dd.MM.yyyy HH:mm:ss'))
    if($q){
        Write-Host ('Очередь: '+$q.state+' | завершено заданий '+$q.completed_jobs+'/'+$q.total_jobs)
        if($q.total_jobs -gt 0){$percent=100*$q.completed_jobs/$q.total_jobs;Write-Host ('Доля заданий: '+$percent.ToString('F1')+'% (не оценка времени всего опыта)')}
        Write-Host ('Текущий этап: '+$q.job.stage+' | вариант: '+$q.job.arm+' | видео: '+$q.job.video)
        if($q.reason){Write-Host $q.reason -ForegroundColor Yellow}
    }
    if($s){Write-Host ('Этап исполнителя: '+$s.stage+' | '+$s.variant);Write-Host ('Последнее обновление: '+$s.updated)}
    $frame=$null;$total=$null
    if($q -and $q.job.stage -eq 'signatures' -and $active.log){
        $path=Join-Path 'E:\Dissertation' $active.log
        $line=Get-Content -LiteralPath $path -Tail 50 -ErrorAction SilentlyContinue | Where-Object {$_ -match '^\[.+\] (\d+) кадров'} | Select-Object -Last 1
        if($line -and $line -match '^\[.+\] (\d+) кадров'){$frame=[int]$Matches[1]}
        $meta=Read-Json ('E:\Dissertation\runs\fly_signatures\'+$q.job.video+'\signatures.json');if($meta){$total=$meta.frames}
    }elseif($s.total -gt 0){$frame=$s.done;$total=$s.total}
    if($null -ne $frame -and $total -gt 0){
        $percent=[Math]::Min(100,100*$frame/$total);Write-Progress -Activity 'Текущий этап' -Status ($frame.ToString()+'/'+$total) -PercentComplete $percent
        Write-Host ('Текущий этап: '+$frame+'/'+$total+' ('+$percent.ToString('F1')+'%)')
    }else{Write-Progress -Activity 'Текущий этап' -Completed;Write-Host 'Объём текущего этапа пока неизвестен.'}
    if($active.started){Write-Host ('Прошло с начала задания: '+([DateTimeOffset]::Now-[DateTimeOffset]::Parse($active.started)).ToString('hh\:mm\:ss'))}elseif($s.elapsed_seconds){Write-Host ('Прошло на этапе: '+$s.elapsed_seconds+' с')}
    $technical=Read-Json (Join-Path $Run 'technical.json');if($technical){Write-Host ('Замер полного объёма по одному train-видео: около '+([double]$technical.rough_full_seconds/3600).ToString('F1')+' ч; оценка неопределённа.')}
    Write-Host 'Осталось до результата: достоверной оценки пока нет.'
    Write-Host ('Сохранение: '+$p.save_at+' | конец только своего дерева: '+$p.hard_stop)
    Write-Host 'Законченные видео, MASA и сшивка сохраняются отдельно. HOTA считается только на всех35.'
    Write-Host 'Закрытие окна не останавливает расчёт и не запускает дубль.'
    if(Test-Path -LiteralPath (Join-Path $Run 'night_finished.json')){break}
    Start-Sleep -Seconds 3
}
Read-Host 'Работа сохранена. Enter — закрыть'
