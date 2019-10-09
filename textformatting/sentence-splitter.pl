#!/usr/bin/env perl
#
# SentenceExtractor was originally part of the WWW2sf package
#
# Split Japanese text to sentences
# usage: echo 'あまり気の毒だから「行く事は行くがじき帰る。来年の夏休みにはきっと帰る」と慰めてやった。明日も。' | perl -I../perl sentence-splitter.pl

use utf8;
use strict;

use SentenceExtractor;

binmode(STDIN, ':utf8');
binmode(STDOUT, ':utf8');

while (<STDIN>) {
    for my $sentence (SentenceExtractor->new($_, 'japanese')->GetSentences()) {
	print $sentence, "\n";
    }
}
